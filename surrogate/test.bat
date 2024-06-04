for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_gat_%%i --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --length 1200 --result_dir r1_flood_gat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_nncat_%%i/20000 --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --result_dir r1_flood_nncat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_gat_%%i --rain_suffix test_bpswmm_ --length 1200 --result_dir r1_flood_gat_%%i_test
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_nncat_%%i/20000 --rain_suffix test_bpswmm_ --result_dir r1_flood_nncat_%%i_test
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_gat_%%i --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --length 1200 --result_dir r1_edgef_flood_gat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_nncat_%%i/20000 --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --result_dir r1_edgef_flood_nncat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_gat_%%i --rain_suffix test_bpswmm_ --length 1200 --result_dir r1_edgef_flood_gat_%%i_test
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_nncat_%%i/20000 --rain_suffix test_bpswmm_ --result_dir r1_edgef_flood_nncat_%%i_test
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_gat_%%i --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --length 1200 --epsilon 0.0 --result_dir r1_gat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_nncat_%%i/20000 --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --epsilon 0.0 --result_dir r1_nncat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_gat_%%i --rain_suffix test_bpswmm_ --length 1200 --epsilon 0.0 --result_dir r1_gat_%%i_test
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_flood_nncat_%%i/20000 --rain_suffix test_bpswmm_ --epsilon 0.0 --result_dir r1_nncat_%%i_test
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_gat_%%i --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --length 1200 --epsilon 0.0 --result_dir r1_edgef_gat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_nncat_%%i/20000 --rain_suffix bpswmm_ --train_event_id train_id_%%i.npy --epsilon 0.0 --result_dir r1_edgef_nncat_%%i_vali
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_gat_%%i --rain_suffix test_bpswmm_ --length 1200 --epsilon 0.0 --result_dir r1_edgef_gat_%%i_test
for /l %%i in (0,1,4) do python main.py --test --model_dir r1_edgef_flood_nncat_%%i/20000 --rain_suffix test_bpswmm_ --epsilon 0.0 --result_dir r1_edgef_nncat_%%i_test