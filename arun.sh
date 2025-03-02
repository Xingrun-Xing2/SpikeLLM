CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model /share/xingxingrun/llama-7b --eval_ppl \
--epochs 20 --output_dir ./log/llama1-7b-w4a4-200 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande --let_lr 1e-3 --alpha 0.75 --seed 2 --addbit 1 --low_p 0.9 > llama-1-7b-0-200.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--model /share/xingxingrun/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log/llama2-7b-w4a4-t2-200 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande --let_lr 1e-3 --alpha 0.75 --seed 2 --addbit 1 --low_p 0.9 > llama-2-7b-t2-200.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
--model /share/xingxingrun/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log/llama2-7b-w4a4-t4-200 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande --let_lr 1e-3 --alpha 0.75 --seed 2 --addbit 2 --low_p 0.95 > llama-2-7b-t4-200.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--model /share/xingxingrun/Llama-2-13b --eval_ppl \
--epochs 20 --output_dir ./log/llama2-13b-w4a4-t2-200 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande --let_lr 1e-3 --alpha 0.75 --seed 2 --addbit 1 --low_p 0.9 > llama-2-13b-t2-200.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python main.py \
--model /share/xingxingrun/Llama-2-13b --eval_ppl \
--epochs 20 --output_dir ./log/llama2-13b-w4a4-t4-200 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande --let_lr 1e-3 --alpha 0.75 --seed 2 --addbit 2 --low_p 0.95 > llama-2-13b-t4-200.log 2>&1 &


