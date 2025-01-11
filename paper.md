Zong Zhi Tan
shauntanzongzhi@u.nus.edu
National University of Singapore


Page 2 and 4 header: Yuyi Zhong, Zong Zhi Tan, Hanping Xu, and Siau-Cheng Khoo



# ABSTRACT


Since the neural network verification problem can be formulated as an optimization problem, linear programming (LP) solvers have been deployed as off-the-shelf tools in the process.

However, existing LP solvers running on CPU scale poorly on large networks.

To expedite the process, we propose an LP-solving theorem specially tailored to neural network verification.

In practice, we transform the constrained solving problem into an unconstrained problem that can be executed on GPU, attaining significant speed-up to the solving process.

We explicitly include constraints on layers that take more than one predecessor instead of handling multiple predecessors by inefficient concatenation.

Our theorem applies to widely used networks, such as fully connected, convolutional, and residual networks.

From the evaluation, our GPU-aided solver achieves precision comparable to that of the state-of-the-art solver GUROBI with significant speed improvement and helps acquire competitive verification precision compared to advanced verification methods.



# 1 INTRODUCTION


Researchers have investigated the verification of neural networks due to their wide application [18, 25, 28].

Throughout the evolution of verification techniques, abstract interpretation-based techniques [6, 15, 20, 22–24, 27, 32] continues to play an important role.

However, due to the nature of over-approximation, the methods could suffer from severe precision loss for deeper networks.

Thus, there is a promising direction for improving abstraction (aka. abstract refinement) with the help of (mixed integer) linear programming (MILP or LP) [21, 31, 34] where GUROBI [11] solver is commonly used despite the scalability concern that it executes on the CPU.


Therefore, we propose a specially tailored theorem for LP solving to speed up the solving process for abstract-refinement-based methods.

It is noteworthy that our theorem could handle all these three kinds of constraints: output constraints, intermediate neuron constraints and constraints of layers that take more than one predecessor, which enhances the scientific rigor of the verification of residual networks.

Our paper offers the methodical transformation from the verification specification to the effective implementation as an analyzer named GRENA (GPU-aided abstract REfinement for Neural network verificAtion), and we assess it against the state-of-the-art tools to empirically support its strong solving and verification capacities.

Our dockerized system, data, usage description and experiment scripts are available at https://github.com/Grena-verifier/Grena-verifier.

We summarize our contributions below:


• We propose a novel, formal and rigorous theorem to solve constrained optimization problems that include output constraints, multi-ReLU constraints, and complex constraints of residual network layers.
• We utilize the multi-ReLU abstraction in WraLU [12] to further tighten our constraint set for precision improvement.
• We offer strong and effective engineering deployments and demonstrate the strong verification efficiency of our system through empirical experiments, while providing our analyzer with detailed demonstration.



# 2 SYSTEM DESIGN


[IMAGE]
Figure 1: The iterative process of abstract refinement.


In general, the verification process repetitively selects adversarial labels and attempts to eliminate them through iterations of refinements as illustrated in Figure 1.

In each iteration, we take the encoding of multiple adversarial labels (the disjunction is handled by following the convention in [34]), the current network abstraction, plus the SOTA WraLU multi-neuron constraints as the constraint set.

We eliminate 𝛿 spurious adversarial labels if the constraint set is infeasible, and eliminating all adversarial labels results in successful verification.

If the constraint set is feasible, we send it to our tailored LP solver on GPU (details deferred till subsection 2.1) and resolve neuron bounds to obtain a refined abstraction, where the refined abstraction is used in the next iteration.

Furthermore, as a feasible constraint set indicates the possibility of a property violation, we collect the batch of input neuron assignments during each solving substep and pass them to the model to check if they constitute an adversarial example so that we can falsify.

We repeat this process until a conclusive result is obtained; or until the time/iteration threshold has exceeded, indicating inconclusive result.



## 2.1 GPU-aided Linear Programming Solver


This subsection presents our theorem of transforming a constrained linear programming problem into an unconstrained solving problem amenable to GPU acceleration.


Preliminaries.

Given a network with 𝐿 + 1 layers and each layer corresponds to a layer index, the input layer is at index 0 and the output layer is at index 𝐿.

We denote a set [𝑅] to include all ReLU layer indexes; the set 𝐿1 contains all linear layer indexes with one connected preceding layer. The set 𝐿2 records all indexes of linear layers that take two preceding layers.

We assume that [𝑅] ∪ [𝐿1] ∪ [𝐿2] = [1, .

.

.

, 𝐿] and both 1, 𝐿 ∈ [𝐿1].

The output and input/preceding layer of a ReLU layer are respectively represented by 𝑥ˆ(𝑖) and 𝑥ˆ(𝑖)𝑝, for 𝑖 ∈ [𝑅].

Given a neuron index 𝑗 and a layer index 𝑖, 𝑥ˆ(𝑖)𝑗 represents the j-th neuron at i-th layer and 𝑥ˆ(𝑖)𝑗𝑝 refers to its input neuron.

Symbol 𝑥(𝑖),𝑖 ∈ [𝐿1] ∪ [𝐿2] represents the output of a linear layer; 𝑥ˆ (0) , 𝑥 (0) both denote the input layer.

Symbol 𝑥 (𝑖) 𝑝 ,𝑖 ∈ [𝐿1] refers to the predecessor of layer 𝑥 (𝑖) for 𝑖 ∈ [𝐿1]; whereas 𝑥 (𝑖) 𝑝1 , 𝑥 (𝑖) 𝑝2 are the two preceding layers of layer 𝑥 (𝑖) for 𝑖 ∈ [𝐿2].

Finally, we designate 𝑆 (𝑖) as a set that includes the indexes of all connected succeeding layers of layer 𝑖 and 𝑖𝑠 ∈ 𝑆 (𝑖); the set 𝑆 2 (𝑖) = ∪𝑖𝑠 ∈𝑆 (𝑖)𝑆 (𝑖𝑠 ), which includes the successors’ indexes of succeeding layers of layer 𝑖 and 𝑖 𝑠 2 ∈ 𝑆 2 (𝑖).


Theorem 1.

The constrained solving problem in neural network verification (as shown in Equation 1) can be transformed into an unconstrained problem in Equation 2 by using Lagrangian dual.


Proof.

The derivation can be found at this appendix


In detail, the constrained problem formulation is given as:


[EQUATION]


In detail, 𝑙(0), 𝑢(0) record the lower and upper bounds of input neurons; 𝐻𝑥 (𝐿) + 𝑑 ≤ 0 represents the output constraints that encode the existence of multiple adversarial examples.

For ReLU neurons, their functionalities depend on the stability statuses.

For example, suppose a linear layer 𝑖 is followed by a ReLU layer 𝑖𝑠.

A ReLU neuron is stably activated if it takes non-negative input interval and thus it equals to the input neuron, and we collect the indexes of those non-negative input neurons at layer 𝑖 as 𝐼 +(𝑖); for the stably deactivated ReLU neurons whose input is non-positive, their outputs are always evaluated to 0, and we denote the indexes of those non-positive input neurons as a set 𝐼 − (𝑖); unstable ReLU neurons take both positive and negative input values, where the input neurons are recorded in 𝐼 ±(𝑖).

In particular, the unstable ReLU neuron is approximated by an orange-colored triangle shape as Figure 2 illustrates, where 𝑙 (𝑖)𝑗 , 𝑢(𝑖)𝑗 record its input interval and 𝑢 (𝑖) 𝑗 𝑢 (𝑖) 𝑗 −𝑙 (𝑖) 𝑗 is abbreviated as 𝑠 (𝑖)𝑗 .

Constraints 𝑃 (𝑖) 𝑥ˆ (𝑖) 𝑝 + 𝑃ˆ(𝑖) 𝑥ˆ (𝑖) −𝑝 (𝑖) ≤ 0 capture the dependencies of multiple ReLU neurons in the same layer, which is obtained from WraLU[12] method to improve solving precision.

The coefficients 𝑐 (0) and 𝑐 (𝑖) ,𝑖 ∈ [𝑅] are used to control the objective function.

As we aim to resolve the input lower and upper bounds of unstable ReLU neurons to refine the abstraction, we only set one element among 𝑐 (0) , 𝑐(𝑖) ,𝑖 ∈ [𝑅] as 1 (for lower bound computation) or -1 (for upper bound) for the respective neuron, the rest of the elements are set as 0.

Eventually, we transform the constrained solving problem to an unconstrained problem using Lagrangian variables as shown below, where we annotate [𝑥]+ = max (𝑥, 0), [𝑥]− = − min (𝑥, 0):


[EQUATION]


Any valid setting of 𝛾, 𝜋 ≥ 0; 𝛼 ∈ [0, 1] leads to a safe lower bound of the original problem.

Based on the values of 𝛾, 𝜋, 𝛼, we compute the values of 𝑣 (𝑖) and 𝑣ˆ (𝑖) in an inverse way from 𝑣 (𝐿) to 𝑣 (0) .

Using all assignments of variables, we could compute the objective value.

In practice, the solving process starts with a valid initialization of 𝛾, 𝜋, 𝛼, then we optimize these variables using gradient information.


Figure 2: The approximation of a ReLU neuron


Algorithm 1 solve_LP_pseudocode


Input:
• 𝑀: neural network model
• L𝐿: list of old lower bounds for all ReLU and input layers
• L𝑈 : list of old upper bounds for all ReLU and input layers
• Π: output constraints; Θ: WraLU constraints


Output: improved lower and upper bounds
1: S ← create_solver_model(𝑀, L𝐿, L𝑈 , Π, Θ)
2: 𝑙𝑖𝑠𝑡_𝑛𝑒𝑤_𝐿,𝑙𝑖𝑠𝑡_𝑛𝑒𝑤_𝑈 ← [], []     ⊲ initialisation
3: for 𝑖 in range(len(L𝐿)) do     ⊲ solve for each layer
4: S.set_layer(𝑖)     ⊲ reset to solve for this layer
5: S.initalise_lagrangian_vars()
6: 𝑚𝑎𝑥_𝑜𝑏 𝑗 ← train_until_convergense(S)
7: 𝑁𝐿, 𝑁𝑈 ← get_new_bounds(L𝐿 [𝑖], L𝑈 [𝑖],𝑚𝑎𝑥_𝑜𝑏 𝑗)     ⊲ improve old bounds based on solved values
8: 𝑙𝑖𝑠𝑡_𝑛𝑒𝑤_𝐿.append(𝑁𝐿)     ⊲ record updated bounds
9: 𝑙𝑖𝑠𝑡_𝑛𝑒𝑤_𝑈 .append(𝑁𝑈 )
10: return 𝑙𝑖𝑠𝑡_𝑛𝑒𝑤_𝐿,𝑙𝑖𝑠𝑡_𝑛𝑒𝑤_𝑈


Algorithm 1 shows the process of solving tighter bounds for each layer by training Lagrangian variables.

Note that the Lagrangian multiplier has been widely used in existing works [5, 9, 28, 30]. To the best of our knowledge, our method is the first time to apply it to spurious-adversarial-label-guided refinement.

Furthermore, we consider multi-neuron constraints, output constraints and 𝐿2 layer constraints that explicitly take two preceding layers, which enhances the theoretical rigor of residual network verification.



# 3 EXPERIMENTS


We compare the performance of our prototypical verifier GRENA with state-of-the-art verifiers including incomplete tool WraLU [12] and the complete tool 𝛼, 𝛽-CROWN [3] - the winner of VNNCOMP (International Verification of Neural Networks Competition).

In addition, we compare our tailored LP solver with SOTA GUROBI with respect to returned bound tightness and execution time.



## 3.1 Experiment Setup


The dataset includes MNIST (denoted as ‘M’) [2] and CIFAR10 (short as ‘C’) [10].

We test over fully-connected (denoted as ‘FC’), convolutional (as ‘Conv’) and residual (’Res’) networks with various sizes, that are obtained from the ERAN system [4] and VNNCOMP [1].

The number of intermediate layers (#Layers), the number of intermediate neurons (#Neurons), and the trained defence are enumerated in Table 1 (a trained defence is a defence method against adversarial examples to improve robustness of networks).



## 3.2 Comparison with SoTA verifiers


To test the verification performance of GRENA, we select 30 images from the datasets for each network to verify robustness and compare the results and time costs.

To verify robustness, we choose a perturbation parameter 𝜖 for each tested network as indicated in Table 1 and apply the perturbation to each image.

We check if all the “perturbed” images within 𝜖 will be classified the same as the original image by the networks as the perturbation is imperceptible to human eyes.

If so, we conclude the robustness to be verified.

Otherwise, we falsify the robustness property if we detect a counterexample with a different label.

If the analysis is inconclusive, we return unknown (abbreviated as ‘#Unk’) to the user.


The verification result of each tool and average execution time per image are shown in Table 2.

We can observe that we win over both WraLU and 𝛼, 𝛽-CROWN with respect to precision as we return more conclusive results (either verified or falsified).

In particular, we return 50.7% more conclusive images than WraLU while WraLU fails to handle two residual networks.

Even compared with the complete tool 𝛼, 𝛽-CROWN, our tool obtains 13 more conclusive images in total and achieve better or the same verification/falsification precision on most networks.

The empirical results indicate the strong verification efficiency of our system.



## 3.3 Comparision with GUROBI


We now compare the bound solving abilities of our tailored solver and GUROBI in the context of neural network encoding.

We select one image for each network and collect all the constraints where we use the constraint set to solve all unstable neuron bounds and input bounds by our solver and GUROBI, later we compare the tightness of the solved bounds as visualized in Figure 3.


Figure 3 are the log-scale histogram plots of bound improvements from GUROBI and our tailored solver, where "improvement" is defined as the original neuron bound minus the new neuron interval returned by the two solvers.

The height of the bars represents the number of neurons with improvement at the magnitude indicated on the x-axis.

We could observe from Figure 3a, 3d, 3e and 3f that the orange bars significantly overlap with the blue bars, which means our tailored solver achieved improvements comparable to that from GUROBI.

It is noteworthy that the average solving time is 35503.32 seconds for GUROBI while only 47.38 seconds for our GPU-accelerated solver, leading to an impressive 749× speedup.


Table 2: The verification results of WraLU, 𝛼, 𝛽-CROWN and our system GRENA with average execution time per image.


Due to space limitation, more results and details can be found in our Github Repo.

In conclusion, our tailored LP solver could obtain comparable bound improvement compared to GUROBI while significantly boosting the solving time.



# 4 RELATED WORK


Generally speaking, verification of deep neural networks is an NP-hard problem [8].

Therefore, there are a series of incomplete verification methods that sacrifice completeness.

Representative works include those abstract interpretation based [6, 14, 19, 20] or bound propagation based [7, 16, 26, 29, 33], etc.

To mitigate the precision loss of incomplete methods, researchers have been relying on LP or MILP to encode the network more tightly.

For example, DeepSRGR [31] or ARENA [34] or PRIMA [17] systems would call up GUROBI solver to resolve LP and obtain tighter neuron intervals.

However, the usage of an off-the-shelf solver on the CPU fails to leverage the nature of neural network encoding.


Figure 3: Bound improvement comparison between our solver and GUROBI.


Inspired by the works that aim to migrate the verification of neural networks to GPUs with the help of Lagrangian dual problems [5, 9, 28], we propose our tailored LP solver on GPU that benefits our LP formulation.

Note that previous works [5, 9, 28] only encode one-predecessor cases where the multiple predecessors would be concatenated into one.

Though it could be handled by other engineering approaches, it lacks rigorous theoretical derivation.

On the contrary, we explicitly encode multi-predecessor cases in our formulation.

Furthermore, [5, 28] only considers intermediate neuron constraints and [9] only includes output constraints in their constraint set; while our formulation captures both intermediate and output constraints.

Lastly, to our knowledge, our method is the first to effectively deploy the Lagrangian dual problem to spurious-adversarial-label-guided refinement process.



# 5 CONCLUSION


In this paper, we propose a theorem to solve for linear programming problem on GPU in the context of neural network verification.

To the best of our knowledge, our work is the first to use Lagrangian dual on spurious-adversarial-label guided refinement process and encode complex network constraints that take more than one predecessor which enhances the scientific rigor of the verification of residual networks.

We implement our solving theorem to a GPU-based tailored solver, and we conduct experiments to indicate that our tailored solver could return comparable solved values compared to GUROBI while obtaining significant speed gain.

Furthermore, it enables our verifier GRENA to return more conclusive results than the SOTA verifiers in a reasonable amount of time, indicating the strong efficacy of our system.
