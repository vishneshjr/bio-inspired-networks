# Introduction

The goal of this project is to study computational networks that have biologically feasible learning mechanisms or structures. Most existing models with biologically feasible features either have very loose analogues to biological realizations or do not perform any useful learning task. How can we make artificial models of neural networks that are biologically plausible while also achieving good performance on some downstream task?

# Idea

There are two approaches I have in mind to explore this research question. The first is a top-down approach where I am first going to start with a simple 3-layer MLP that has good performance on the MNIST dataset (Implementation similar to the one in [this simple book](http://neuralnetworksanddeeplearning.com)). Then I want to modify existing connections and feedback signals to mimic more biologically feasible methods and see if I can do so while maintaining good performance. My initial intuitions are to try ideas mentioned in [these](https://www.pnas.org/doi/10.1073/pnas.1912804117) [papers.](https://www.nature.com/articles/s41583-020-0277-3)
The second approach is a bottom-up approach where I begin with implementations of networks resembling biological neurons like [this](https://www.sciencedirect.com/science/article/pii/S0896627321005018?via%3Dihub) and adapt it to have good performance on some learning task.

# Notes

I am very open to any feedback/resources/interesting ideas to try out. You can reach me at vishnesh@gatech.edu

