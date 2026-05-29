# Export per-node .nsys-rep -> sqlite, run INSIDE the alps3 container so the
# nsys reader version matches the writer (the login node has no nsys, and
# host copies are older than the report and refuse to open it).
#   args: $1 = directory holding node*.nsys-rep   $2 = output dir for *.sqlite
#         $3.. = node names to export (default: node0 node4)
# See export.sbatch for how this is submitted.
REP=${1:?usage: run_export.sh <rep_dir> <out_dir> [node0 node4 ...]}
WORK=${2:?usage: run_export.sh <rep_dir> <out_dir> [node0 node4 ...]}
shift 2
NODES=${*:-node0 node4}
NSYS=$(command -v nsys)
echo "[in-container] host=$(hostname)  nsys=$NSYS"
$NSYS --version
mkdir -p "$WORK"
for n in $NODES; do
  echo "=== exporting $n @ $(date +%H:%M:%S) ==="
  $NSYS export --type sqlite --force-overwrite true \
     -o "$WORK/$n.sqlite" "$REP/$n.nsys-rep"
  echo "  exit=$? @ $(date +%H:%M:%S)"
  ls -la "$WORK/$n.sqlite"
done
echo "ALL_DONE"
