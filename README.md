# [ACML 2024] Countering Relearning with Perception Revising Unlearning

This is a PyTorch implementation of [Countering Relearning with Perception Revising Unlearning](https://github.com/DATA-Transpose/PRU) accepted by ACML 2024.

## Abstract

Unlearning methods that rely solely on forgetting data typically modify the network’s decision boundary to achieve unlearning. However, these approaches are susceptible to the ``relearning" problem, whereby the network may recall the forgotten class upon subsequent updates with the remaining class data. Our experimental analysis reveals that, although these modifications alter the decision boundary, the network's fundamental perception of the samples remains mostly unchanged. In response to the relearning problem, we introduce the Perception Revising Unlearning (PRU) framework. PRU employs a probability redistribution method, which assigns new labels and more precise supervision information to each forgetting class instance. The PRU actively shifts the network's perception of forgetting class samples toward other remaining classes. The experimental results demonstrate that PRU not only has good classification effectiveness but also significantly reduces the risk of relearning, suggesting a robust approach to class unlearning tasks that depend solely on forgetting data.

## Environment Setting

Install the environment and install all necessary packages:

```bash
conda env create -f environment.yaml
```

## Results Reproduce:  

Run code blocks in the `prepare_data.ipynb` to prepare the data.

Run `train_original.py` to train the original model.

Run `train_unlearn.py` to perform unlearning.

Run `train_relearn.py` to perform the relearning test with only remaining data.

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Citation

```
@inproceedings{  

}
```