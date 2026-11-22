# Earth-2 RS · eventos severos no Rio Grande do Sul

Public page: https://itamariliuk.github.io/earth2-rs-eventos-severos/

*English below.*

Previsões retrospectivas (*hindcasts*) com o **FourCastNet 3**, um dos modelos de IA do NVIDIA Earth-2, para dois
eventos no Rio Grande do Sul: o temporal da madrugada de 20/11/2026 em Tupanciretã e a enchente de maio de 2024.
Cada rodada tem 24 membros (24 futuros possíveis a partir do mesmo começo), para dar probabilidade em vez de um
palpite único.

**O que o ensemble mostrou**

- Temporal de 20/11/2026: chance de 80 a 100 % de atmosfera instável (Índice K > 32) sobre o centro-norte do RS já
  na previsão feita em 15/11, 4,8 dias antes. Chance de 40 a 60 % de mais de 10 mm em 6 h em Tupanciretã.
- Enchente de maio de 2024, previsão feita em 26/04: o modelo acertou a região (centro-norte do estado), mas
  subestimou o volume: a média chegou a cerca de 120 mm, contra cerca de 270 mm no ERA5. Ainda assim, 20 a 40 % dos
  membros passaram de 150 mm.

**Limites**

- Grade de 25 km: o modelo não resolve tornado nem a rajada de um temporal (a rajada máxima prevista sobre o RS
  ficou em ~70 km/h). Ele mostra o *ambiente* de risco.
- Os modelos regionais do Earth-2 (StormCast, CorrDiff) só existem hoje para EUA, Europa e Taiwan.
- Pesquisa em simulação; não é um sistema de alerta nem substitui a meteorologia oficial e a Defesa Civil.

**Como reproduzir** (pasta `code/`)

1. Numa máquina com GPU de 80 GB (usamos uma H100 PCIe): `bash setup_gpu.sh`. A receita foi reconstruída a partir
   da instalação que funcionou e explica, em comentários, os passos que deram problema.
2. `bash run_all.sh` roda as cinco rodadas, uma de cada vez (pouco mais de uma hora). Os NetCDF (4,4 GB) não estão
   no repositório.
3. As figuras não precisam de GPU; os comandos estão no fim de `run_all.sh`. `gerar_anim_k.ps1` gera a animação
   no Windows, num ambiente temporário do `uv`.

| Arquivo | O que faz |
|---|---|
| `run_fcn3_rs.py` | Ensemble FCN3 + WindgustAFNO + PrecipitationAFNO v1 + Índice K; salva a caixa da América do Sul |
| `plot_rs.py` | Mapas de probabilidade sobre o RS, uma coluna por rodada |
| `plot_extra.py` | Meteograma de Tupanciretã, comparação de maio de 2024 com o ERA5 e as animações |
| `check_nc.py` | Conferência rápida de um NetCDF (NaN, faixas, divergência entre membros) |

Condições iniciais: GFS (NOAA, arquivo na AWS) para novembro de 2026 e ERA5 (Copernicus/ECMWF, via ARCO) para
maio de 2024, baixadas pelo Earth2Studio 0.20.0. Os pesos dos modelos são baixados pelo Earth2Studio e seguem as
licenças dos próprios modelos.

---

## English

Hindcasts with **FourCastNet 3**, one of the NVIDIA Earth-2 AI models, for two events in Rio Grande do Sul,
Brazil: the severe storm of November 20, 2026 in Tupanciretã and the May 2024 flood. Each run has 24 members.

- November 20, 2026 storm: 80–100 % chance of an unstable atmosphere (K-index > 32) over north-central Rio Grande
  do Sul already in the forecast made on November 15, 4.8 days ahead; 40–60 % chance of more than 10 mm in 6 h at
  Tupanciretã.
- May 2024 flood, forecast made on April 26: right region, volume too low (ensemble mean about 120 mm against
  about 270 mm in ERA5), yet 20–40 % of members exceeded 150 mm.
- Limits: a 25 km grid does not resolve tornadoes or storm gusts; it shows the risk environment. Earth-2 regional
  models (StormCast, CorrDiff) currently cover only the US, Europe and Taiwan. Simulation research, not a warning
  system.

Reproduce with `code/setup_gpu.sh` and `code/run_all.sh` on an 80 GB GPU; plotting needs no GPU.

---

Developed at LABRIOT (Laboratório de Robótica e IoT), UTFPR Ponta Grossa, Brazil, as part of Dr. Itamar Iliuk's
activities as an NVIDIA Omniverse Ambassador. LABRIOT is a member of the NVIDIA Inception program. NVIDIA, Earth-2
and FourCastNet are trademarks of NVIDIA; this project is not endorsed by NVIDIA.
