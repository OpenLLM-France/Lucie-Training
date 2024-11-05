set -e

ASSET_FOLDER=`realpath $0`
ASSET_FOLDER=`dirname $ASSET_FOLDER`
ASSET_FOLDER=`dirname $ASSET_FOLDER`
ASSET_FOLDER=$ASSET_FOLDER/assets/hugging_face

CONFIG=$ASSET_FOLDER/config.json
README=$ASSET_FOLDER/README.md

LUCIE_MODEL_FOLDER="/lustre/fsn1/projects/rech/qgz/commun/trained_models/Lucie/pretrained"

TOKENIZER_FOLDER="/lustre/fsn1/projects/rech/fwx/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer"
if [ ! -d $TOKENIZER_FOLDER ];then
    TOKENIZER_FOLDER=$HOME/models/Lucie/tokenizer
fi

REPO_BASE="OpenLLM-France/Lucie-7B"

for UPLOAD_OPTIMIZER in 0 1;do

    if [ $UPLOAD_OPTIMIZER -eq 0 ];then
        echo "Upload checkpoints"
        REPO_NAME=$REPO_BASE
        PARENT_FOLDER="$LUCIE_MODEL_FOLDER/transformers_checkpoints"
        if [ -d $PARENT_FOLDER ];then
            STEPS=`ls -d $PARENT_FOLDER/global_step*000 | awk -F "step" '{print $NF}' | sort -nr `
        else
            STEPS=""
            i=5000
            while [ $i -lt 25000 ];do
                STEPS="$i $STEPS"
                i=$((i+5000))
            done
            while [ $i -lt 500000 ];do
                STEPS="$i $STEPS"
                i=$((i+25000))
            done
        fi
    else
        echo "Upload optimizer states"
        if [ $UPLOAD_OPTIMIZER -eq 1 ];then
            PARENT_FOLDER="$LUCIE_MODEL_FOLDER/universal_checkpoints"
            REPO_NAME=$REPO_BASE"-optimizer-states"
        else
            PARENT_FOLDER="$LUCIE_MODEL_FOLDER/checkpoints"
            REPO_NAME=$REPO_BASE"-optimizer-states-512GPU"
        fi
        STEPS=""
        i=25000
        while [ -d "$PARENT_FOLDER/global_step$i" ];do
            STEPS="$i $STEPS"
            i=$((i+25000))
        done
    fi

    ############################################
    # 1. Upload tokenizer and minimal stuff

    if [ $UPLOAD_OPTIMIZER -eq 0 ];then

        UPLOAD_OPTION=""

        python3 hf_upload_model.py $REPO_NAME \
            $TOKENIZER_FOLDER \
            --message "Upload tokenizer" # --add_files_in_folder true --message "Upload tokenizer, model config and minimal README"

        UPLOAD_OPTION="--type checkpoint"

    else

        UPLOAD_OPTION="--type optimizer"

        rm -Rf empty
        mkdir empty

        python3 hf_upload_model.py $REPO_NAME $UPLOAD_OPTION \
            empty \
            --message "Upload minimal README and config" # --add_files_in_folder true --message "Upload tokenizer, model config and minimal README"

        rm -Rf empty

    fi

    ############################################
    # 2. Upload checkpoints

    for STEP in $STEPS;do
        FOLDER="$PARENT_FOLDER/global_step$STEP"

        # Do not upload again what's already on Hugging Face
        # Ugly: to be modified
        if [ $UPLOAD_OPTIMIZER -eq 0 ];then
            if [ $STEP -le 700000 ];then
                continue
            fi
        else
            if [ $STEP -le 700000 ];then
                continue
            fi
        fi

        if [ ! -d $FOLDER ];then
            echo "ERROR: $FOLDER does not exist"
            # Hack : need to have an empty folder to update config/READMEs in revisions
            exit 0
            FOLDER=empty
        fi

        # Upload model folder to Hugging Face
        python3 hf_upload_model.py $REPO_NAME $UPLOAD_OPTION \
            $FOLDER \
            --training_steps $STEP \
            --message "Upload checkpoint at step $STEP" \

    done

    # # TODO at the end ...
    # python3 hf_upload_model.py $REPO_NAME $UPLOAD_OPTION \
    #     $TOKENIZER_FOLDER \
    #     --training_steps -1 \
    #     --message "Update README" \

done