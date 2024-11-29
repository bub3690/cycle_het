python train.py --dataset edema --epochs 50 --batch_size 32 --arch cnn --backbone resnet18 --optimizer Adam  --tsne
python train.py --dataset edema --epochs 50 --batch_size 32 --arch cnn --backbone resnet34 --optimizer Adam --tsne
python train.py --dataset edema --epochs 50 --batch_size 32 --arch cnn --backbone resnet50 --optimizer Adam --tsne

python train.py --dataset edema --epochs 50 --batch_size 32 --arch hybrid --backbone resnet18 --optimizer Adam  --tsne
python train.py --dataset edema --epochs 50 --batch_size 32 --arch hybrid --backbone resnet34 --optimizer Adam --tsne
python train.py --dataset edema --epochs 50 --batch_size 32 --arch hybrid --backbone resnet50 --optimizer Adam --tsne

python train.py --dataset ralo --epochs 50  --batch_size 32 --arch cnn --backbone resnet18 --optimizer Adam --tsne
python train.py --dataset ralo --epochs 50  --batch_size 32 --arch cnn --backbone resnet34 --optimizer Adam --tsne
python train.py --dataset ralo --epochs 50  --batch_size 32 --arch cnn --backbone resnet50 --optimizer Adam --tsne

python train.py --dataset ralo --epochs 50  --batch_size 32 --arch hybrid --backbone resnet18 --optimizer Adam --tsne
python train.py --dataset ralo --epochs 50  --batch_size 32 --arch hybrid --backbone resnet34 --optimizer Adam --tsne
python train.py --dataset ralo --epochs 50  --batch_size 32 --arch hybrid --backbone resnet50 --optimizer Adam --tsne


python train.py --dataset inha --epochs 50  --batch_size 32 --arch cnn --backbone resnet18 --optimizer Adam --tsne
python train.py --dataset inha --epochs 50  --batch_size 32 --arch cnn --backbone resnet34 --optimizer Adam --tsne
python train.py --dataset inha --epochs 50  --batch_size 32 --arch cnn --backbone resnet50 --optimizer Adam --tsne

python train.py --dataset inha --epochs 50  --batch_size 32 --arch hybrid --backbone resnet18 --optimizer Adam --tsne
python train.py --dataset inha --epochs 50  --batch_size 32 --arch hybrid --backbone resnet34 --optimizer Adam --tsne
python train.py --dataset inha --epochs 50  --batch_size 32 --arch hybrid --backbone resnet50 --optimizer Adam --tsne



