# PhysicsInformedML
Physics Informed Machine Learning for Astrophysics

## Introduction
### Machine Learning in Science
Modern machine learning is becoming increasingly important in science. We collect increasingly larger amounts of experimental data, but our ability to analyze this data has not kept up with the data avalanche. Machine learning, in particular variants of Deep Learning are emerging as a promising way to overcome this barrier. However, in science we need to understand and estimate the statistical significance of our derived results, and there is a general skepticism towards ‘black-box’ techniques. For data with large dimensions, the networks can get quite large, making training slow and cumbersome. As a result, serious attention is being given to Physics Informed Machine Learning – how we can use prior knowledge about underlying symmetries, geometric and physical properties of the data to simplify network design.

Another important property of nature is sparsity -- most natural phenomena can be well represented with a very small set of parameters. This has led to the notion of compressed sensing and shrinkage estimators in statistics.  Autoencoders, and variational autoencoders have been used successfully to find the latent dimensions, and recently sparse convolutional neural networks capable of sparsifying the network structure have been also advocated, 

My thesis work is in (1) applying these ideas to the analysis of medium resolution stellar spectra, (2) developing Deep Learning algorithms for stellar parameter inference of stars observed, and (3) designing AI telescope algorithm -- reinforcement learning on target selections that maximizes scientific outputs -- for our team at JHU and Princeton.

These tasks represents a recently emerging direction in AI, where instead of controlling self-driving cars, the AI is helping us to run our scientific instruments more optimally. This ultimate goal can break down into several sub-problems. A collection of these will be thoroughly explained in the sections below. 

### DATA: the PFS Survey
Motivated by the discovery potential of wide-field surveys of the sky, the world-wide astronomical community is undertaking a series of comprehensive surveys using the next generation of wide-field optical telescopes, imaging cameras, multi-object spectrographs and software pipelines. 

The Prime Focus Spectrograph (PFS) is a major new multi-object fiber-fed spectrograph currently under construction for the Subaru 8.2-meter telescope on the summit of Mauna Kea, Hawai’i (one of the largest telescopes in the world). PFS will be the largest and most powerful multi-object spectrograph in the world when it sees first light in early 2022. It has 2394 optical fibers simultaneously deployed over a field of view of 1.3 deg$^2$. The spectral range is 0.38 to 1.26 microns with a resolution of about 3A.

The astronomers building this \$80M instrument are designing a survey, to take 300 nights of telescope time.
The project relevant to my thesis work is studying the makeup of the Milky Way Galaxy and its companions. Mapping the distribution, motions, and chemical composition of stars in the major components of our own home galaxy and its neighbor, the Andromeda galaxy, plus smaller satellite systems, will constrain models of galaxy formation, insights that can be directly compared with what we learn from the study of distant galaxies.

One of the great challenges of designing the survey to address these questions is determining exactly which astronomical objects will be observed spectroscopically, and how long each should be exposed. An on-going imaging survey using a camera called Hyper Suprime-Cam (HSC), also on the Subaru telescope, is providing the raw material for the PFS: it is identifying the faint stars and galaxies of which PFS will obtain spectra.

However, there are thousand times more objects in the imaging data than we can obtain spectra for in 300 nights. We therefore have to carefully select the optimal targets. Here we describe our planned approach, with a focus on new techniques combining astrophysics, computer science and machine learning.

We seek to optimize the target selection for PFS such that we can maximize the scientific utility of the entire survey given its observational constraints. This task is complicated by our not knowing in advance what the data will show. We therefore seek to devise a scheme that allows us to update the target selection on-the-fly such that we get an optimal overall yield. The science cases of PFS are complex and multi-pronged, and one of the challenges of optimizing the targeting for them is that the utilities (i.e., the science return from them as a function of the nature of the data) are highly non-linear functions that may not be fully known in advance.

To illustrate the challenge, consider the situation in which, after a total of three hours of exposure time, 60% of the stars targeted in a given field will have spectra of sufficient quality to measure their temperature and some chemical abundances, of which half are good enough to measure the makeup of the star,  How much more exposure time should be given to the remaining stars?  How do we balance that relative to the utility of other targets in the field that have not yet been observed? This is a coupled optimization problem which requires understanding of the multiple simultaneous scientific goals to be addressed, and the information content associated with additional data on each object. This requires using insights from machine learning, astrophysics, and computer science to develop and refine our target selection algorithms, both in the initial list of targets, and the almost real-time decisions that will be made as the observations are being carried out. 


