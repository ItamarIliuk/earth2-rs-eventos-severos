# Gera os quadros da animação com o Índice K (modo animk do plot_extra.py) no notebook.
# 1) baixa Space Grotesk e Inter do repositório oficial do Google Fonts para ~/fonts
# 2) roda o script num ambiente temporário do uv (não altera nenhum Python instalado)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$fonts = Join-Path $HOME "fonts"
New-Item -ItemType Directory -Force $fonts | Out-Null
$base = "https://raw.githubusercontent.com/google/fonts/main/ofl"
$lista = @{
  "SpaceGrotesk-wght.ttf"     = "$base/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf"
  "Inter-opsz-wght.ttf"       = "$base/inter/Inter%5Bopsz,wght%5D.ttf"
}
foreach ($nome in $lista.Keys) {
  $dest = Join-Path $fonts $nome
  if (-not (Test-Path -LiteralPath $dest)) { Invoke-WebRequest -Uri $lista[$nome] -OutFile $dest }
}
uv run --no-project --python 3.13 --with xarray --with netcdf4 --with cartopy --with matplotlib --with numpy `
  python plot_extra.py animk nc/fcn3_rs_2026-11-15T00_gfs.nc
