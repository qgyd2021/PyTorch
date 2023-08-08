#!/usr/bin/env bash

# nohup sh run.sh --system_version centos --stage -1 --stop_stage 2 &

# sh run.sh --system_version windows --stage -1 --stop_stage -1

# params
system_version="centos";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

num_nodes=1
node_rank=0

pretrained_bert_model_name=chinese-bert-wwm-ext

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
pretrained_models_dir="${work_dir}/../../pretrained_models";

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

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download spacy model"

  python3 -m spacy download en_core_web_sm
  python3 -m spacy download de_core_news_sm

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: make vocabulary"

  python3 1.make_vocabulary.py \
  --src_vocab_pkl "${data_dir}/vocab_de.pkl" \
  --tgt_vocab_pkl "${data_dir}/vocab_en.pkl"

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: train model"

  python3 2.train_model.py \
  --src_vocab_pkl "${data_dir}/vocab_de.pkl" \
  --tgt_vocab_pkl "${data_dir}/vocab_en.pkl"

fi
