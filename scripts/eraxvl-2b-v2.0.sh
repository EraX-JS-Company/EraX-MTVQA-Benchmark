CUDA_VISIBLE_DEVICES=0 python -m eval.qwen2vl \
--image_folder /paht/to/folder/Vi-MTVQA/MTVQA/test/imgs/VI \
--output_folder results \
--bench_file /paht/to/folder//Vi-MTVQA/MTVQA/test/json/test_VI.json \
--model_path /paht/to/model \
--save_name <save file name> \
--num_workers 30 \
--batch_size 4
