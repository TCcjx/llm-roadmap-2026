#!/usr/bin/env bash
# 批量检查仓库内所有 http(s) 链接的存活状态。
#
# 用法:
#   bash scripts/check_links.sh                    # 检查全部 markdown
#   bash scripts/check_links.sh README.md 03-*.md  # 只检查指定文件
#   REPORT=report.md bash scripts/check_links.sh   # 同时写一份报告
#
# 判定分级:
#   2xx/3xx           → OK
#   401/403/429       → BLOCKED（反爬/需登录，通常不是失效）
#   404/410/451       → DEAD（真失效）→ 退出码 1
#   000/5xx（重试后） → ERROR（网络受限或服务端故障）→ 退出码 2
#
# 可用环境变量:
#   TIMEOUT=20 RETRIES=2 PARALLEL=8 REPORT=path
#
# 国内环境先设代理:
#   export https_proxy=http://127.0.0.1:7890 http_proxy=$https_proxy

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
  mapfile -t FILES < <(find "$ROOT" -name '*.md' -not -path '*/.git/*' -not -path '*/node_modules/*')
fi

TIMEOUT="${TIMEOUT:-20}"
RETRIES="${RETRIES:-2}"
PARALLEL="${PARALLEL:-8}"
REPORT="${REPORT:-}"
UA="Mozilla/5.0 (compatible; llm-roadmap-link-checker/1.0)"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# ---------- 抽取链接 ----------
# 先去反引号，再按字符类排掉空格/括号/引号，最后去掉行尾标点
cat "${FILES[@]}" 2>/dev/null \
  | tr -d '\000' \
  | tr '`' ' ' \
  | grep -oE "https?://[^ )<>\"']+" \
  | sed -E "s/[.,;:!?\"']+$//" \
  | grep -vE '^(https?://127\.0\.0\.1|https?://localhost)' \
  | sort -u > "$work/urls.txt"

total=$(wc -l < "$work/urls.txt" | tr -d ' ')
echo "待检查链接: ${total} 条（超时 ${TIMEOUT}s，重试 ${RETRIES}，并发 ${PARALLEL}）"
echo

# ---------- 探测 ----------
probe() {
  url="$1"
  attempt=1
  code=000
  while [ "$attempt" -le "$RETRIES" ]; do
    code=$(curl -sS -o /dev/null -L --max-time "$TIMEOUT" \
      -A "$UA" --ssl-no-revoke \
      -w '%{http_code}' "$url" 2>/dev/null) || code="000"
    case "$code" in
      2??|3??) break ;;
    esac
    attempt=$((attempt+1))
    [ "$attempt" -le "$RETRIES" ] && sleep 1
  done
  printf '%s\t%s\n' "$code" "$url"
}
export -f probe
export TIMEOUT RETRIES UA

tr '\n' '\0' < "$work/urls.txt" | xargs -0 -n 1 -P "$PARALLEL" bash -c 'probe "$1"' _ > "$work/results.tsv"

# ---------- 分类 ----------
awk -F'\t' '{
  c = $1
  if (c ~ /^[23]/)                    print "OK\t" $2
  else if (c ~ /^(401|403|429)$/)     print "BLOCKED\t" $2
  else if (c ~ /^(404|410|451)$/)     print "DEAD\t" $2
  else                                print "ERROR\t" $2
}' "$work/results.tsv" | sort > "$work/classified.tsv"

print_report() {
  echo "== 汇总 =="
  cut -f1 "$work/classified.tsv" | sort | uniq -c | awk '{printf "  %-8s %s\n", $2, $1}'
  echo
  for kind in DEAD ERROR BLOCKED; do
    n=$(grep -c "^${kind}	" "$work/classified.tsv" 2>/dev/null || true)
    [ "${n:-0}" -eq 0 ] && continue
    echo "== ${kind}（${n} 条）=="
    grep "^${kind}	" "$work/classified.tsv" | cut -f2 | sed 's/^/  /'
    echo
  done
  echo "说明：BLOCKED 多为反爬/需登录，通常不是真失效；ERROR 多为网络受限或服务端 5xx。"
}

if [ -n "$REPORT" ]; then
  print_report > "$REPORT"
fi
print_report

dead=$(grep -c "^DEAD	" "$work/classified.tsv" 2>/dev/null || true)
errs=$(grep -c "^ERROR	" "$work/classified.tsv" 2>/dev/null || true)

echo
echo "结论：DEAD=${dead:-0}  ERROR=${errs:-0}  总计 ${total} 条"
[ "${dead:-0}" -gt 0 ] && exit 1
[ "${errs:-0}" -gt 0 ] && exit 2
exit 0