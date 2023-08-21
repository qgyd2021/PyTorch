#!/usr/bin/env bash

# nohup sh run.sh --stage -1 --stop_stage 2 --system_version centos &

# sh run.sh --system_version windows --stage -1 --stop_stage -1

# params
system_version="centos";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

pretrained_bert_model_name=bert-base-uncased
patience=3


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
data_dir="$(pwd)/data_dir"
pretrained_models_dir="${work_dir}/../../../pretrained_models";

mkdir -p "${data_dir}"
mkdir -p "${pretrained_models_dir}"

vocabulary_dir="${data_dir}/vocabulary"
serialization_dir="${data_dir}/serialization_dir"

export PYTHONPATH="${work_dir}/../../.."

if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/PyTorch/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  source /data/local/bin/PyTorch/bin/activate
  alias python3='/data/local/bin/PyTorch/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  source /data/local/bin/PyTorch/bin/activate
  alias python3='/data/local/bin/PyTorch/bin/python3'
fi


declare -A pretrained_bert_model_dict
pretrained_bert_model_dict=(
  ["chinese-bert-wwm-ext"]="https://huggingface.co/hfl/chinese-bert-wwm-ext"
  ["bert-base-uncased"]="https://huggingface.co/bert-base-uncased"
  ["bert-base-japanese"]="https://huggingface.co/cl-tohoku/bert-base-japanese"
  ["bert-base-vietnamese-uncased"]="https://huggingface.co/trituenhantaoio/bert-base-vietnamese-uncased"
)
pretrained_model_dir="${pretrained_models_dir}/${pretrained_bert_model_name}"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download pretrained model"

  if [ ! -d "${pretrained_model_dir}" ]; then
    mkdir -p "${pretrained_models_dir}"
    cd "${pretrained_models_dir}" || exit 1;

    repository_url="${pretrained_bert_model_dict[${pretrained_bert_model_name}]}"
    git clone "${repository_url}"

    cd "${pretrained_model_dir}" || exit 1;
    rm flax_model.msgpack && rm pytorch_model.bin && rm tf_model.h5
    wget "${repository_url}/resolve/main/pytorch_model.bin"
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 1.make_vocabulary.py \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --vocabulary_dir "${vocabulary_dir}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: train model"
  cd "${work_dir}" || exit 1;

  python 2.train_model.py \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --serialization_dir "${serialization_dir}" \
  --patience "${patience}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: test model"
  cd "${work_dir}" || exit 1;

  python3 3.test_model.py \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --weights_file "${serialization_dir}/best.th" \

fi
