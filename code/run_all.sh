#!/usr/bin/env bash
# As cinco rodadas, uma de cada vez: cada processo FCN3 ocupa ~68 GB da GPU.
# Numa H100 PCIe o conjunto leva pouco mais de uma hora. Rodar de dentro de code/:  bash run_all.sh
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16   # sem limite, os threads de CPU disputam a máquina e a GPU fica parada
mkdir -p nc
M=24
run() { [ -f "nc/$5" ] && { echo "pula $4"; return; }; uv run python run_fcn3_rs.py --init $1 --end $2 --members $M --source $3 --out "nc/$5" > "run_$4.log" 2>&1; }
run 2026-11-15T00 2026-11-21T00 gfs  nov15   fcn3_rs_2026-11-15T00_gfs.nc
run 2026-11-16T00 2026-11-21T00 gfs  nov16   fcn3_rs_2026-11-16T00_gfs.nc
run 2026-11-17T00 2026-11-21T00 gfs  nov17   fcn3_rs_2026-11-17T00_gfs.nc
run 2026-11-18T00 2026-11-21T00 gfs  nov18   fcn3_rs_2026-11-18T00_gfs.nc
run 2024-04-26T00 2024-05-03T00 arco mai2024 fcn3_rs_2024-04-26T00_arco.nc
echo ALL_DONE

# Figuras (não precisam de GPU):
#   uv run python plot_rs.py --event-start 2026-11-19T18 --event-end 2026-11-20T12 nc/fcn3_rs_2026-11-1*_gfs.nc
#   uv run python plot_extra.py meteo nc/fcn3_rs_2026-11-1*_gfs.nc
#   uv run python plot_extra.py maio  nc/fcn3_rs_2024-04-26T00_arco.nc
#   uv run python plot_extra.py animk nc/fcn3_rs_2026-11-15T00_gfs.nc
#   ffmpeg -framerate 3 -i figs/anim_k/q%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 figs/fcn3_rs_anim_k.mp4
