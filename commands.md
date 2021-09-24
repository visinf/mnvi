## Commands

### UCI Regression

To train and evaluate MNVI models on boston:
```
python uci.py --datasetname=boston --lr=0.05 --batch_size=64 --prior_precision=1e-1
```

To train and evaluate MNVI models on concrete:
```
python uci.py --datasetname=concrete --lr=0.05 --batch_size=64 --prior_precision=1e-1
```

To train and evaluate MNVI models on energy:
```
python uci.py --datasetname=energy --lr=0.05 --batch_size=64 --prior_precision=1e-1
```

To train and evaluate MNVI models on kin8:
```
python uci.py --datasetname=kin8 --lr=0.05 --batch_size=128 --prior_precision=1e-1
```

To train and evaluate MNVI models on power:
```
python uci.py --datasetname=power --lr=0.05 --batch_size=128 --prior_precision=1e-1
```

To train and evaluate MNVI models on wine:
```
python uci.py --datasetname=wine --lr=0.05 --batch_size=128 --prior_precision=1e-1
```

To train and evaluate MNVI models on yacht:
```
python uci.py --datasetname=yacht --lr=0.05 --batch_size=64 --prior_precision=1e-2
```

To train and evaluate SMFVI models on boston:
```
python uci.py --datasetname=boston --lr=0.05 --batch_size=64 --prior_precision=1e-1 --net=MFSVI
```

To train and evaluate SMFVI models on concrete:
```
python uci.py --datasetname=concrete --lr=0.05 --batch_size=64 --prior_precision=1e-1 --net=MFSVI
```

To train and evaluate SMFVI models on energy:
```
python uci.py --datasetname=energy --lr=0.05 --batch_size=64 --prior_precision=1e-1 --net=MFSVI
```

To train and evaluate SMFVI models on kin8:
```
python uci.py --datasetname=kin8 --lr=0.05 --batch_size=128 --prior_precision=1e-1 --net=MFSVI
```

To train and evaluate SMFVI models on power:
```
python uci.py --datasetname=power --lr=0.05 --batch_size=128 --prior_precision=1e-1 --net=MFSVI
```

To train and evaluate SMFVI models on wine:
```
python uci.py --datasetname=wine --lr=0.05 --batch_size=128 --prior_precision=1e-1 --net=MFSVI
```

To train and evaluate SMFVI models on yacht:
```
python uci.py --datasetname=yacht --lr=0.05 --batch_size=64 --prior_precision=1e-2 --net=MFSVI
```


### Image Classification Training

To train LeNet MFVI on MNIST:
```
python main.py --batch_size=64 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.1 --lr_scheduler_milestones="[10]" --model=LeNetMFVI --model_kl_div_weight=5e-7 --model_prior_precision=1e3 --model_min_variance=1e-5 --num_workers=4 --optimizer=Adam --optimizer_lr=0.001 --optimizer_amsgrad=True  --total_epochs=20 --training_dataset=MnistTrain --training_dataset_root=data/mnist --training_key=total_loss --validation_dataset=MnistValid --validation_dataset_root=data/mnist --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/LeNetMFVI-TrainMnsit-001 --checkpoint=None --seed=1
```

To train LeNet SMFVI on MNIST:
```
python main.py --batch_size=64 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.1 --lr_scheduler_milestones="[10]" --model=LeNetMFSVI --model_kl_div_weight=5e-7 --model_prior_precision=1e3 --model_min_variance=1e-5 --num_workers=4 --optimizer=Adam --optimizer_lr=0.001 --optimizer_amsgrad=True  --total_epochs=20 --training_dataset=MnistTrain --training_dataset_root=data/mnist --training_key=total_loss --validation_dataset=MnistValid --validation_dataset_root=data/mnist --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/LeNetMFSVI-TrainMnsit-001 --checkpoint=None --seed=1
```

To train LeNet MNVI on MNIST:
```
python main.py --batch_size=64 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.1 --lr_scheduler_milestones="[10]" --model=LeNetMNVI --model_kl_div_weight=2e-7 --model_prior_precision=1e3 --model_min_variance=1e-5 --model_mnv_init=-3.0 --num_workers=4 --optimizer=Adam --optimizer_lr=0.001 --optimizer_amsgrad=True  --total_epochs=20 --training_dataset=MnistTrain --training_dataset_root=data/mnist --training_key=total_loss --validation_dataset=MnistValid --validation_dataset_root=data/mnist --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/LeNetMNVI-TrainMnsit-001 --checkpoint=None --seed=1
```

To train AllConvNet MFVI on CIFAR-10:
```
python main.py --batch_size=128 --loss=ClassificationLossVI  --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.1 --lr_scheduler_milestones="[150, 200, 225]" --model=AllConvNetMFVI --model_kl_div_weight=2e-8 --model_min_variance=1e-5 --model_prior_precision=1e3 --num_workers=4 --optimizer=SGD --optimizer_lr=0.05 --optimizer_nesterov=True  --optimizer_momentum=0.9 --total_epochs=250 --training_dataset=Cifar10Train --training_dataset_root=data/cifar10 --training_key=total_loss --validation_dataset=Cifar10Valid --validation_dataset_root=data/cifar10 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/AllConvNetMFVI-TrainCIFAR-001 --checkpoint=None --seed=1
```

To train AllConvNet SMFVI on CIFAR-10:
```
python main.py --batch_size=128 --loss=ClassificationLossVI  --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.1 --lr_scheduler_milestones="[150, 200, 225]" --model=AllConvNetMFSVI --model_kl_div_weight=2e-8 --model_min_variance=1e-5 --model_prior_precision=1e3 --num_workers=4 --optimizer=SGD --optimizer_lr=0.05 --optimizer_nesterov=True  --optimizer_momentum=0.9 --total_epochs=250 --training_dataset=Cifar10Train --training_dataset_root=data/cifar10 --training_key=total_loss --validation_dataset=Cifar10Valid --validation_dataset_root=data/cifar10 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/AllConvNetMFSVI-TrainCIFAR-001 --checkpoint=None --seed=1
```

To train AllConvNet MNVI on CIFAR-10:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.1 --lr_scheduler_milestones="[150, 200, 225]" --model=AllConvNetMNVI --model_kl_div_weight=2e-8 --model_min_variance=1e-5 --model_prior_precision=1e4 --model_mnv_init=-3.0 --num_workers=4 --optimizer=SGD --optimizer_lr=0.05 --optimizer_nesterov=True  --optimizer_momentum=0.9  --total_epochs=250 --training_dataset=Cifar10Train --training_dataset_root=data/cifar10 --training_key=total_loss --validation_dataset=Cifar10Valid --validation_dataset_root=data/cifar10 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/AllConvNetMNVI-TrainCIFAR-001 --checkpoint=None --seed=1
```

To train ResNet18 MFVI on CIFAR-10:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.2 --lr_scheduler_milestones="[60, 120, 160]" --model=ResNet18MFVI --model_kl_div_weight=5e-8  --model_num_classes=10 --model_min_variance=1e-5 --model_prior_precision=1e4 --num_workers=4 --optimizer=SGD --optimizer_lr=0.1 --optimizer_nesterov=True --optimizer_momentum=0.9 --total_epochs=200 --training_dataset=Cifar10Train --training_dataset_root=data/cifar10 --training_key=total_loss --validation_dataset=Cifar10Valid --validation_dataset_root=data/cifar10 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/ResNetMFVI-TrainCIFAR10-001 --checkpoint=None --seed=1
```

To train ResNet18 SMFVI on CIFAR-10:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.2 --lr_scheduler_milestones="[60, 120, 160]" --model=ResNet18MFSVI --model_kl_div_weight=5e-8  --model_num_classes=10 --model_min_variance=1e-5 --model_prior_precision=1e4 --num_workers=4 --optimizer=SGD --optimizer_lr=0.1 --optimizer_nesterov=True --optimizer_momentum=0.9 --total_epochs=200 --training_dataset=Cifar10Train --training_dataset_root=data/cifar10 --training_key=total_loss --validation_dataset=Cifar10Valid --validation_dataset_root=data/cifar10 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/ResNetMFSVI-TrainCIFAR10-001 --checkpoint=None --seed=1
```

To train ResNet18 MNVI on CIFAR-10:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.2 --lr_scheduler_milestones="[60, 120, 160]" --model=ResNet18MNVI --model_kl_div_weight=5e-8  --model_num_classes=10 --model_min_variance=1e-5 --model_mnv_init=-3.0 --model_prior_precision=1e4 --num_workers=4 --optimizer=SGD --optimizer_lr=0.1 --optimizer_nesterov=True --optimizer_momentum=0.9 --total_epochs=200 --training_dataset=Cifar10Train --training_dataset_root=data/cifar10 --training_key=total_loss --validation_dataset=Cifar10Valid --validation_dataset_root=data/cifar10 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/ResNetMNVI-TrainCIFAR10-001 --checkpoint=None --seed=1
```

To train ResNet18 MFVI on CIFAR-100:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.2 --lr_scheduler_milestones="[60, 120, 160]" --model=ResNet18MFVI --model_kl_div_weight=2e-8 --model_num_classes=100 --model_min_variance=1e-5 --model_prior_precision=1e4 --num_workers=4 --optimizer=SGD --optimizer_lr=0.1 --optimizer_nesterov=True --optimizer_momentum=0.9 --total_epochs=200 --training_dataset=Cifar100Train --training_dataset_root=data/cifar100 --training_key=total_loss --validation_dataset=Cifar100Valid --validation_dataset_root=data/cifar100 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/ResNetMFVI-TrainCIFAR100-001 --checkpoint=None --seed=1
```

To train ResNet18 SMFVI on CIFAR-100:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.2 --lr_scheduler_milestones="[60, 120, 160]" --model=ResNet18MFSVI --model_kl_div_weight=2e-8 --model_num_classes=100 --model_min_variance=1e-5 --model_prior_precision=1e4 --num_workers=4 --optimizer=SGD --optimizer_lr=0.1 --optimizer_nesterov=True --optimizer_momentum=0.9 --total_epochs=200 --training_dataset=Cifar100Train --training_dataset_root=data/cifar100 --training_key=total_loss --validation_dataset=Cifar100Valid --validation_dataset_root=data/cifar100 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/ResNetMFSVI-TrainCIFAR100-001 --checkpoint=None --seed=1
```

To train ResNet18 MNVI on CIFAR-100:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.2 --lr_scheduler_milestones="[60, 120, 160]" --model=ResNet18MNVI --model_kl_div_weight=5e-8 --model_num_classes=100 --model_min_variance=1e-5 --model_mnv_init=-3.0 --model_prior_precision=1e4 --num_workers=4 --optimizer=SGD --optimizer_lr=0.1 --optimizer_nesterov=True --optimizer_momentum=0.9 --total_epochs=200 --training_dataset=Cifar100Train --training_dataset_root=data/cifar100 --training_key=total_loss --validation_dataset=Cifar100Valid --validation_dataset_root=data/cifar100 --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]" --device=cuda:0 --save=output/ResNetMNVI-TrainCIFAR100-001 --checkpoint=None --seed=1
```

To train ResNet18 MNVI on ImageNet:
```
python main.py --batch_size=128 --loss=ClassificationLossVI --lr_scheduler=MultiStepLR --lr_scheduler_gamma=0.1 --lr_scheduler_milestones="[20, 40]" --model=ImageResNet18MNVI --model_kl_div_weight=2e-9 --model_num_classes=1000 --model_min_variance=1e-5 --model_mnv_init=-3.0 --model_prior_precision=1e5 --num_workers=12  --optimizer=SGD --optimizer_nesterov=True --optimizer_momentum=0.9 --optimizer_lr=0.05  --total_epochs=60 --training_dataset=ImageNetTrain --training_dataset_root=data/imagenet/train --training_key=total_loss --validation_dataset=ImageNetValid --validation_dataset_root=data/imagenet/val --validation_keys="[top1,xe]" --validation_keys_minimize="[False,True]"  --device=cuda_parallel --save=output/ImageResNet18MNVI-Train-001 --checkpoint=None --seed=1
```

### Image Classification Evaluation

To evaluate LeNet MFVI on MNIST:
```
python evaluate_uncertainty.py --filename=lenet_mfvi_mnist.ckpt --net=LeNet --mode=MFVI --dataset=MnistTest --dataset_path=data/mnist --device=cuda:0
```

To evaluate LeNet SMFVI on MNIST:
```
python evaluate_uncertainty.py --filename=lenet_mfsvi_mnist.ckpt --net=LeNet --mode=MFSVI --dataset=MnistTest --dataset_path=data/mnist --device=cuda:0
```

To evaluate LeNet MNVI on MNIST:
```
python evaluate_uncertainty.py --filename=lenet_mnvi_mnist.ckpt --net=LeNet --mode=MNVI --dataset=MnistTest --dataset_path=data/mnist --device=cuda:0
```

To evaluate AllConvNet MFVI on CIFAR-10:
```
python evaluate_uncertainty.py --filename=allcnn_mfvi_cifar10.ckpt --net=AllConvNet --mode=MFVI --dataset=Cifar10Test --dataset_path=data/cifar10 --device=cuda:0
```

To evaluate AllConvNet SMFVI on CIFAR-10:
```
python evaluate_uncertainty.py --filename=allcnn_mfsvi_cifar10.ckpt --net=AllConvNet --mode=MFSVI --dataset=Cifar10Test --dataset_path=data/cifar10 --device=cuda:0
```

To evaluate AllConvNet MNVI on CIFAR-10:
```
python evaluate_uncertainty.py --filename=allcnn_mnvi_cifar10.ckpt --net=AllConvNet --mode=MNVI --dataset=Cifar10Test --dataset_path=data/cifar10 --device=cuda:0
```

To evaluate ResNet18 MFVI on CIFAR-10:
```
python evaluate_uncertainty.py --filename=resnet18_mfvi_cifar10.ckpt --net=ResNet18 --mode=MFVI --num_classes=10 --dataset=Cifar10Test --dataset_path=data/cifar10 --device=cuda:0
```

To evaluate ResNet18 SMFVI on CIFAR-10:
```
python evaluate_uncertainty.py --filename=resnet18_mfsvi_cifar10.ckpt --net=ResNet18 --mode=MFSVI --num_classes=10 --dataset=Cifar10Test --dataset_path=data/cifar10 --device=cuda:0
```

To evaluate ResNet18 MNVI on CIFAR-10:
```
python evaluate_uncertainty.py --filename=resnet18_mnvi_cifar10.ckpt --net=ResNet18 --mode=MNVI --num_classes=10 --dataset=Cifar10Test --dataset_path=data/cifar10 --device=cuda:0
```

To evaluate ResNet18 MFVI on CIFAR-100:
```
python evaluate_uncertainty.py --filename=resnet18_mfvi_cifar100.ckpt --net=ResNet18 --mode=MFVI --num_classes=100 --dataset=Cifar100Test --dataset_path=data/cifar100 --device=cuda:0
```

To evaluate ResNet18 SMFVI on CIFAR-100:
```
python evaluate_uncertainty.py --filename=resnet18_mfsvi_cifar100.ckpt --net=ResNet18 --mode=MFSVI --num_classes=100 --dataset=Cifar100Test --dataset_path=data/cifar100 --device=cuda:0
```

To evaluate ResNet18 MNVI on CIFAR-100:
```
python evaluate_uncertainty.py --filename=resnet18_mnvi_cifar100.ckpt --net=ResNet18 --mode=MNVI --num_classes=100 --dataset=Cifar100Test --dataset_path=data/cifar100 --device=cuda:0
```

To evaluate ResNet18 MNVI on ImageNet:
```
python evaluate_uncertainty.py --filename=resnet18_mnvi_imagenet.ckpt --net=ImageResNet18 --mode=MNVI --num_classes=1000 --dataset=ImageNetValid --dataset_path=data/imagenet/valid --device=cuda:0
```