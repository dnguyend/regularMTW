# Folder contains example workbooks and numerical testing of the results in the paper.
* DemoSimpleMTW.ipynb: demos of classes in simple_mtw.
  * * For GenHyperbolicSimpleMTW, where $s = p_0e^{p_1u} + p_2e^{p_3u}$, depending on the sign combinations, we have 26 subcases, where the $\sinh$ case has a global inverse, while the other cases have restrictions on range and several branches.
  * * The Lambert case has two branches. The Exponential-Trigonometric case has infinitely many branches.
    * We test the inverse function, metric compatibility and the curvature formula.
* MTW_tensor_for_SphereAndHyperboloid.ipynb: demos for space_form.py, and symbolic verification of the formulas in the proof of theorems in section 4.
    * Numerically check that the proposed costs in the paper satisfy A3(sw)
* SINHHomogeneousPower.ipynb: Check the result on absolutely homogeneous convex function of order $\alpha > 1$
    * Note that the inverse of the optimal map is usually solved numerically.
    * We check numerically the supremum is satisfied, using an optimizer in jaxopt.
    * Check the formulas for the optimal map and its inverse 
      * When $x$ is in range $\phi(x) <\frac{1}{\alpha r}$ the supremum is attainable
      * Otherwise it is not attainable and the optimizer could only get close to the proposed value. We check statistically it gets close but cannot exceed the values proposed in the theorem.
* SINHHomogeneousOne.ipynb: Check the result on absolutely-homogeneous convex functions of order $\alpha = 1$. Similar to the above.

          
