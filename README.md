# regularMTW
Project for new examples satisfying Ma-Trudinger-Wang condition

The folder src contains the main library for the examples.
* The file src/simple_mtw.py contains the main class.
  * baseSimpleMTW is the main base class. The metric, connection and curvature are implemented if we derive from it, providing the function s and its derivatives, and its inverse $u$.
  * class GenHyperbolicSimpleMTW(baseSimpleMTW) Derived class, for $s$ of the form $p_0e^{p_1u}+ p_2e^{p_3u}$
  * class LambertSimpleMTW(baseSimpleMTW) Derived class, for $s = (a_0+ a_1e^{a_2u}$
  * class TrigSimpleMTW(baseSimpleMTW Derived class, for $s = b_0e^{b_1u}\sin(b_2u+b_3)$
* The file src/space_form.py contains the class for a pseudosphere. The ambient symmetric bilinear form is given by the matrix $A$ in the constructor. THe two main cases are $A = I_n$ (sphere) and $A=(-1, 1,\cdots ,1)$ (hyperboloid model of hyperbolic geometry), but most functions work for an arbitrary symmetric invertible matrix.

# Folder colab contains the colab example/demo workbooks.
    
  
