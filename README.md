# ReID_Adversarial_Defense
Thank you for your attention, this code is for  A Person Re-identification Data Augmentation Method with Adversarial Defense Effect(https://arxiv.org/abs/2101.08783)

The latest version of the paper 'Robust Person Re-identification with Multi-Modal Joint Defence' can be viewed at http://arxiv.org/abs/2111.09571 (The complete code for the new version of the paper will be open sourced after the paper is published)

By providing our code, you can verify the validity of the method proposed in this paper. The github links of the strong baseline can be found in the paper(the dataset can be downloaded from github)

Follow the steps below to verify:
[1] In the ‘reid-strong-baseline/data/transforms/’directory of the strong baseline code file, add the code "multimodal.py" provided by the supplementary material.

[2]Then open ‘build.py’ in the directory mentioned in [1] and write "from .multimodal import *" on line 10(The code file involved in this instructions can be opened online through github to determine the location of the addition described here).

[3] In line 21, if you write “LGPR(0.4)”, you can verify the LGPR method; if you write “T.RandomGrayscale(0.05)”, you can verify the GGPR method; if you write “Fuse_RGB_Gray_Sketch()”, you can verify the Multi-Modal Defense method.



Just in case, due to torch version issues, you may also need to replace lines 174-179 in "reid-strong-baseline/modeling/baseline.py" with the following code:

  def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            if 'classifier' in k:
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])



[4] Change the "MAX_EPOCHS" in line 28 of the "reid-strong-baseline/configs/softmax_triplet_with_center.yml" file to 480 (set a larger value and then observe the training log to get the weight of the optimal epoch, generally only about 320 epochs are required)



Refer to the following commands to train the model and test:

Train:

python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml'  MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('./logs/market_test_epoch320')"


Test：
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('./logs/MultiModal_epoch480/resnet50_model_400.pth')"

#############################################################

Adversarial Defense Experiment:

strong baseline provides trained model weights on github, which can be used for comparative experiments adversarial defense. Since the weight file is relatively large and web pointers are not allowed, we cannot provide our trained weight files for verification, this requires training by adding "Fuse_RGB_Gray_Sketch()" through step [3]. Then use the trained adversarial samples we provided for testing: directly replace the adversarial sample set with the query set in the original Market-501 dataset, and then run the test command. The adversarial samples we provide are "query_Metric-Attack"(https://drive.google.com/file/d/1eZ-ePSFz_X77Z6UUupPb2euUxgz4R0Sb/view?usp=sharing) and "query_MS-SSIM-Attack_epoch30"(https://drive.google.com/file/d/12dD6ZZ5BtxJz3pQ5kZa35jXWe7FPbdaE/view?usp=sharing).
When testing multi-modal defense, you should also add “T.Resize([100, 50])” or “T.Resize([110, 50])” to lines 25-26 of ‘reid-strong-baseline/data/transforms/build.py’
-----------------------------------------------------------------------------------------------------------------------------------------------------

