set -e

ASSET_FOLDER=`realpath $0`
ASSET_FOLDER=`dirname $ASSET_FOLDER`
ASSET_FOLDER=`dirname $ASSET_FOLDER`
ASSET_FOLDER=$ASSET_FOLDER/assets/hugging_face

CONFIG=$ASSET_FOLDER/config.json
README=$ASSET_FOLDER/README.md

TOKENIZER_FOLDER="/lustre/fsn1/projects/rech/fwx/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer"
if [ ! -d $TOKENIZER_FOLDER ];then
    TOKENIZER_FOLDER=$HOME/models/Lucie/tokenizer
fi


PARENT_FOLDER="/lustre/fsn1/projects/rech/qgz/commun/trained_models/Lucie/pretrained/transformers_checkpoints"
if [ -d $PARENT_FOLDER ];then
    STEPS=`ls $PARENT_FOLDER/global_step*000 | awk -F "step" '{print $NF}' | sort -n `
else
    i=5000
    while [ $i -lt 25000 ];do
        STEPS="$STEPS $i"
        i=$((i+5000))
    done
    while [ $i -lt 500000 ];do
        STEPS="$STEPS $i"
        i=$((i+25000))
    done
fi

# Done the first time, then it shouldn't change anything
python3 hf_upload_model.py OpenLLM-France/Lucie-7B \
    $TOKENIZER_FOLDER \
    --message "Upload tokenizer, model config and minimal README" \

for STEP in $STEPS;do
    FOLDER="$PARENT_FOLDER/global_step$STEP"

    if [ ! -d $FOLDER ];then
        # Hack : need to have an empty folder to update config/READMEs in revisions
        FOLDER=empty
    fi	

    # Upload model folder to Hugging Face
    python3 hf_upload_model.py OpenLLM-France/Lucie-7B \
        $FOLDER \
        --training_steps $STEP \
        --is_checkpoint true \
        --message "Upload checkpoint at step $STEP" \

done

python3 hf_upload_model.py OpenLLM-France/Lucie-7B \
    $TOKENIZER_FOLDER \
    --training_steps -1 \
    --message "Upload README" \
