#!/usr/bin/env bash

# nohup sh run.sh --system_version centos --stage 1 --stop_stage 2 &

# sh run.sh --system_version windows --stage -1 --stop_stage -1

# params
system_version="centos";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5


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

mkdir -p "${data_dir}"

train_subset="${data_dir}/train.jsonl"
valid_subset="${data_dir}/valid.jsonl"
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
  $verbose && echo "stage -1: download data"
  cd "${data_dir}" || exit 1;

  area_code_array=(
    52
    91
    92
    234
    254
    255
    886
  )
  for area_code in ${area_code_array[*]}
  do
    if [ ! -d "${area_code}" ]; then
      wget "https://huggingface.co/datasets/qgyd2021/early_media/resolve/main/data/segmented/${area_code}.zip"
      unzip "${area_code}.zip"
      rm -rf "${area_code}.zip"
    fi
  done

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --data_path "${data_dir}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 2.make_vocabulary.py \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train model"
  cd "${work_dir}" || exit 1;

  python3 3.train_model.py \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --serialization_dir "${serialization_dir}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: test model"
  cd "${work_dir}" || exit 1;

  python3 4.test_model.py \
  --vocabulary_dir "${vocabulary_dir}" \
  --weights_file "${serialization_dir}/best.th" \
  --trace_model_filename "${data_dir}/trace_model.zip" \
  --trace_quantized_model_filename "${data_dir}/trace_quantized_model.zip" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: test jit model"
  cd "${work_dir}" || exit 1;

  python3 5.test_jit_model.py \
  --vocabulary_dir "${vocabulary_dir}" \
  --trace_model_filename "${data_dir}/trace_model.zip" \
  --trace_quantized_model_filename "${data_dir}/trace_quantized_model.zip" \

fi
