---
layout: post
title: Experiment on Bi-fidelity Bayesian surrogate model
featured-img: bifidel_GPvsNN_cover
image: bifidel_GPvsNN
category: [bayesian, regression, multi-fidelity]
mathjax: true
summary: Experiment with the Bi-fidelity Bayesian method on a toy model using Neural Network Ensemble (bagging and random prior) and compare it with Gaussian Process
---

The cost of acquiring enough **high-fidelity (HF)** data from the simulations or experiments can be daunting. When much cheaper but less accurate **low-fidelity (LF)** data is available, [multi-fidelity modeling](https://mlatcl.github.io/mlphysical/lectures/05-02-multifidelity.html) methods augment the limited **HF** data with cheaply-obtained **LF** approximations.

The Bayesian paradigm provides a coherent approach for specifying sophisticated hierarchical models: The **HF** data (*evidence*) update the *posterior* (our target) model conditioned on the *prior* model (*belief*) that is constructed from **LF** data. 

# 1. Toy Model

We use the following toy model:

$$ y = A x^2 - B \exp{ \left(\sum_i^n c_i \cos (w_i x-b_i)\right)} $$

where the parameters are arbitrarily chosen such that the **HF** and **LF** are slightly different. We consider a scenario that the **LF** toy model is not very different from the **HF** toy model.
![]({{ "assets/img/bifidel_GPvsNN/toymodel.jpg" | absolute_url }})

We also consider a scenario that the cost of **LF** data acquisition is less than one-tenth of the cost of the **HF** data acquisition. For example, the cost of acquiring 20 high-fidelity data is more expensive than the cost of acquiring 10 high-fidelity data and 100 low-fidelity data. In this regard, the data shown in the plot are randomly chosen 200 points for **LF** and 20 for **HF**.


# 2. Gaussian Process (GP)


### 2.1 Single Fidelity GP
Let us start with the GP modeling on the **HF** data. To be fair, we used 40 **HF** data ( instead of 20 **HF** data ) to construct the GP surrogate model ( we used RBF kernel).
![]({{ "assets/img/bifidel_GPvsNN/high-fidelity-GP.jpg" | absolute_url }})
observe that where the data points are scarce, large model uncertainty (*epistemic uncertainty*) is present. 


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

Observe that in some of regions where the data points are scarce, there can be large discrepancy b/w **LF** and **HF** GP model.


# 3. NN Bagging (Bootstrap aggregating)

Although GP is exact Bayesian, the compuational complexity renders it impractical (without approximation) for high-dimensional problem. Here, we present ensemble neural network method to construct the bi-fidelity bayesian surrogate model. Specifically, we use boostraped data to train multiple NN. Furthermore, each NN is equipped with random prior to make it Bayesian. 

### 3.1 Single fidelity NN Bagging

Here, we used 40 **HF** data, as we did for single fideulity GP.
![]({{ "assets/img/bifidel_GPvsNN/high-fidelity-baggingNN.jpg" | absolute_url }})

### 3.1 Low fidelity NN Bagging

Using 200 **LF** data,
![]({{ "assets/img/bifidel_GPvsNN/low-fidelity-baggingNN.jpg" | absolute_url }})

### 3.2 Bi fidelity NN Bagging

Using 20 **HF** and 200 **LF** data,
![]({{ "assets/img/bifidel_GPvsNN/linear-bi-fidelity-baggingNN.jpg" | absolute_url }})



# References

* [Multifidelity Modelling](https://mlatcl.github.io/mlphysical/lectures/05-02-multifidelity.html)
* [Randomized Prior Functions for Deep Reinforcement Learning](https://papers.nips.cc/paper/8080-randomized-prior-functions-for-deep-reinforcement-learning.pdf): shows the shortcomings with other techniques and motivates the use of bootstrap and prior functions
