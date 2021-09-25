---
layout: post
title: Experiment on Bi-fidelity Bayesian surrogate model
featured-img: bifidel_GPvsNN_cover
image: bifidel_GPvsNN
category: [bayesian, regression, multi-fidelity]
mathjax: true
summary: Experiment with the Bi-fidelity method on a toy model using (Bayesian) Bootstrap aggregating Neural Network and compare it with Gaussian Process
---


# Problem statement
The cost of acquiring enough **high-fidelity (HF)** data from the heavy simulations or experiments can be daunting. When much cheaper but less accurate **low-fidelity (LF)** data is available, multi-fidelity modeling <sup>[1](https://mlatcl.github.io/mlphysical/lectures/05-02-multifidelity.html)</sup> methods augment the limited **HF** data with cheaply-obtained **LF** approximations.

The Bayesian paradigm provides a coherent approach for specifying sophisticated hierarchical models: The **HF** data (*evidence*) update the *posterior* (our target) model conditioned on the *prior* model (*belief*) that is constructed from **LF** data. In this post, I demonstrate Bi-fidelity modeling performance on a toy model using Bayesian Neural-Network ensemble and compare it with the Gaussian Process. 

In addition, Bayesian paradigm propagates uncertaintyies between fideilities. This is very useful information for surrogate model based decision making.



# 1. Toy Model

Let the toy model be in the following form:

$$ y = A x^2 - B \exp{ \left(\sum_i^n c_i \cos (w_i x-b_i)\right)} $$

where the parameters are arbitrarily chosen such that the **HF** and **LF** are slightly different as shown below:
![]({{ "assets/img/bifidel_GPvsNN/toymodel.jpg" | absolute_url }})

Throughout this post, we consider a scenario that the cost of **LF** data acquisition is less than one-tenth of the cost of the **HF** data acquisition. For example, the cost of acquiring 20 high-fidelity data is more expensive than the cost of acquiring 10 high-fidelity data and 100 low-fidelity data. In this regard, the data shown in the plot are randomly chosen 200 points for **LF** and 20 for **HF**. Note that all the data is noiselss, i.e., well aligned with the ground truth toy model. Therefore, we expect that a good surrogate model to well capture the model uncertainty ([*epistemic uncertainty*](https://link.springer.com/article/10.1007/s10994-021-05946-3)) not the data uncertainty (i.e., *risk* or *aleatoric* uncertainty).


# 2. Gaussian Process (GP)


### 2.1 Single Fidelity GP
Here, single fidelity GP is trained on the **HF** data. To be fair, I used 40 **HF** data ( instead of 20 **HF** data ) to train GP. The [RBF kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) is assumed and it's hyper parameters are optimized for maximum likelihood of data. Following plot shows the result.
![]({{ "assets/img/bifidel_GPvsNN/high-fidelity-GP.jpg" | absolute_url }})

Observe that where the data points are few, GP predicts large uncertainty. Note also that the **HF** ground truth is well within prediced undertainty.


### 2.2 Linear bi-fidelity GP

One of the widely used methods of multi-fidelity modeling is to assume that the **HF** and **LF** functions are related by

$$
f_H(x) = f_{\text{err}}(x) + \rho \,f_L(x)
$$

where $f_{\text{err}}(x)$ and $f_L(x)$ are assumed to be independent GP and $\rho$ is a scalar scaling factor. This can be a good approach when $f_H(x)$ depends linearly on $f_L(x)$.  Using 20 **HF** and 200 **LF** data, the linear bi-fidelity GP resulted in:
![]({{ "assets/img/bifidel_GPvsNN/linear-bi-fidelity-GP.jpg" | absolute_url }})



### 2.3 Nonlinear bi-fidelity GP

The linear approach will fail when $f_H(x)$ and $f_L(x)$ are nonlinearly related. In this case, the nonlinear multi-fidelity GP assumes the following:

$$
f_H(x) = f_{\text{err}}(x) + \rho (\,f_L(x) )
$$

This approach is called deep GP in analogous with deep neural network. Again, Using 20 **HF** and 200 **LF** data, the nonlinear bi-fidelity GP resulted in:

![]({{ "assets/img/bifidel_GPvsNN/nonlinear-bi-fidelity-GP.jpg" | absolute_url }})

Although, nonlinear assumption between the fildelities is more general than the linear assumption, the result above is disappointing in a sense that it is over-confident. In other words, the uncertainty prediction did not cover the true **HF** curve. This may ascribed to the fact that the two fideilities of our toy model are very close each other linearly. 


# 3. Bootstrap aggregating (*Bagging*) Neural-Network (NN)

Although GP is exact Bayesian, the compuational complexity renders it impractical (without approximation) for high-dimensional problem. Here, we use ensemble neural network method to construct the bi-fidelity bayesian surrogate model. 
The principle of ensembling for uncertainty quantification follows from the fact that:
	- Each NN tend to converge near the data points but vary over regions (of input domain) where data is absent.
Specifically, I use *Bagging*<sup>[2](https://www.stat.berkeley.edu/~breiman/bagging.pdf)</sup> over NN. Using different boostrapped data to train each NN further helps to avoid overfit. 


### 3.1 Single fidelity *Bagging* NN 

For performance comparison of bi-fidelity model, here, we used 40 **HF** data to train single (high) fidelity model.
![]({{ "assets/img/bifidel_GPvsNN/high-fidelity-baggingNN.jpg" | absolute_url }})

### 3.1 Low fidelity *Bagging* NN 

Using 200 **LF** data,
![]({{ "assets/img/bifidel_GPvsNN/low-fidelity-baggingNN.jpg" | absolute_url }})

### 3.2 Bi fidelity Bagging NN

Using 20 **HF** and 200 **LF** data,
![]({{ "assets/img/bifidel_GPvsNN/linear-bi-fidelity-baggingNN.jpg" | absolute_url }})



# References

* 1. [Multifidelity Modelling](https://mlatcl.github.io/mlphysical/lectures/05-02-multifidelity.html)
* 2. [Bagging](https://www.stat.berkeley.edu/~breiman/bagging.pdf) 
* 3. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf): arguments that dropout (at test time) in NNs has a connection to gaussian processes and motivates its usage as a bayesian method
* 4. [Risk versus Uncertainty in Deep Learning: Bayes, Bootstrap and the Dangers of Dropout](http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf): motivates that dropout with fixed $p$ estimates risk and not uncertainty

