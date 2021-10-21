# PhysicsInformedML
Physics Informed Machine Learning for Astrophysics
- Build better spectra generator with high order interpolator
- Build physics informed autoencoders for (denoising, feature pruning)
- Develope hierarchical schemes to estimate astrophysical parameters from stellar spectra.
- Apply these AI tools to aid Bayesian Hierarchical Modeling
- Design AI telescope algorithm -- reinforcement learning on target selections that maximizes scientific outputs

## Introduction
### Machine Learning in Science
Modern machine learning is becoming increasingly important in science. We collect increasingly larger amounts of experimental data, but our ability to analyze this data has not kept up with the data avalanche. Machine learning, in particular variants of Deep Learning are emerging as a promising way to overcome this barrier. However, in science we need to understand and estimate the statistical significance of our derived results, and there is a general skepticism towards ‘black-box’ techniques. For data with large dimensions, the networks can get quite large, making training slow and cumbersome. As a result, serious attention is being given to Physics Informed Machine Learning – how we can use prior knowledge about underlying symmetries, geometric and physical properties of the data to simplify network design.

Another important property of nature is sparsity -- most natural phenomena can be well represented with a very small set of parameters. This has led to the notion of compressed sensing and shrinkage estimators in statistics.  Autoencoders, and variational autoencoders have been used successfully to find the latent dimensions, and recently sparse convolutional neural networks capable of sparsifying the network structure have been also advocated, 

These tasks represents a recently emerging direction in AI, where instead of controlling self-driving cars, the AI is helping us to run our scientific instruments more optimally. This ultimate goal can break down into several sub-problems. A collection of these will be thoroughly explained in the sections below. 

### DATA: the PFS Survey
Motivated by the discovery potential of wide-field surveys of the sky, the world-wide astronomical community is undertaking a series of comprehensive surveys using the next generation of wide-field optical telescopes, imaging cameras, multi-object spectrographs and software pipelines, with scientific goals that include the structure of our Milky Way Galaxy; the evolution of galaxies; and the distribution and properties of dark matter. One of the great challenges of designing the survey to address these questions is determining exactly which astronomical objects should be observed in detail and how long each should be exposed. As the telescope costs $100k to operate per night, we must carefully select the 500k objects out of billions to target. Our research is to develop deep learning algorithms that optimize this target selection with insights from physics, statistics, data science and AI. 




