#!/usr/bin/env bash

# nohup sh run.sh --stage 0 --stop_stage 1 --system_version centos &
# sh run.sh --stage 0 --stop_stage 1 --system_version windows
# sh run.sh --stage 0 --stop_stage 0 --system_version centos
# sh run.sh --stage 1 --stop_stage 1 --system_version centos
# sh run.sh --stage 1 --stop_stage 2 --system_version centos
# sh run.sh --stage -1 --stop_stage 1

# bitsandbytes
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

pretrained_model_name=gpt2-chinese-cluecorpussmall

normalize_file=normalize.txt

train_subset=train.jsonl
valid_subset=valid.jsonl

final_model_name=gpt2_chinese_h_novel

checkpoint_name=final

# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done

$verbose && echo "system_version: ${system_version}"

work_dir="$(pwd)"
file_dir="$(pwd)/file_dir"
pretrained_models_dir="${work_dir}/../../../pretrained_models";
serialization_dir="${file_dir}/serialization_dir"
final_model_dir="${work_dir}/../../../trained_models/${final_model_name}";

mkdir -p "${file_dir}"
mkdir -p "${pretrained_models_dir}"
mkdir -p "${serialization_dir}"
mkdir -p "${final_model_dir}"


export PYTHONPATH="${work_dir}/../../.."


if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/Transformers/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  # conda activate Transformers
  alias python3='/usr/local/miniconda3/envs/Transformers/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  # conda activate Transformers
  alias python3='/usr/local/miniconda3/envs/Transformers/bin/python3'
fi


declare -A pretrained_model_dict
pretrained_model_dict=(
  ["gpt2-chinese-cluecorpussmall"]="https://huggingface.co/uer/gpt2-chinese-cluecorpussmall"
  ["gpt2"]="https://huggingface.co/gpt2"
  ["japanese-gpt2-medium"]="https://huggingface.co/rinna/japanese-gpt2-medium"

)
pretrained_model_dir="${pretrained_models_dir}/${pretrained_model_name}"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download pretrained model"
  cd "${file_dir}" || exit 1;

  if [ ! -d "${pretrained_model_dir}" ]; then
    cd "${pretrained_models_dir}" || exit 1;

    repository_url="${pretrained_model_dict[${pretrained_model_name}]}"
    git clone "${repository_url}"

    cd "${pretrained_model_dir}" || exit 1;
    rm flax_model.msgpack && rm pytorch_model.bin && rm tf_model.h5
    wget "${repository_url}/resolve/main/pytorch_model.bin"
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: text normalize prepare"
  cd "${work_dir}" || exit 1;

  python3 1.test_normalize.py \
  --output_file "${file_dir}/${normalize_file}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: prepare data"
  cd "${work_dir}" || exit 1;

  python3 2.prepare_data.py \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train model"
  cd "${work_dir}" || exit 1;

  python3 3.train_model.py \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \
  --pretrained_model_name_or_path "${pretrained_models_dir}/${pretrained_model_name}" \
  --output_dir "${serialization_dir}"

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: collect files"
  cd "${work_dir}" || exit 1;

  cp "${serialization_dir}/${checkpoint_name}/pytorch_model.bin" "${final_model_dir}/pytorch_model.bin"

  cp "${pretrained_models_dir}/${pretrained_model_name}/config.json" "${final_model_dir}/config.json"
  cp "${pretrained_models_dir}/${pretrained_model_name}/special_tokens_map.json" "${final_model_dir}/special_tokens_map.json"
  cp "${pretrained_models_dir}/${pretrained_model_name}/tokenizer_config.json" "${final_model_dir}/tokenizer_config.json"
  cp "${pretrained_models_dir}/${pretrained_model_name}/vocab.txt" "${final_model_dir}/vocab.txt"

fi
