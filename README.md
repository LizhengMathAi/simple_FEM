<h1><b>Introduction</b></h1>
The finite element method (FEM) is the most widely used method for solving problems of engineering and mathematical models.
Typical problem areas of interest include the traditional fields of structural analysis, heat transfer, fluid flow, mass transport, and electromagnetic potential.
The FEM is a particular numerical method for solving partial differential equations.<br>
<font color="red">
However, traditional relevant softwares are too large to analyze their structures build-in,
So, I construct a super lightweight FEM framework with highly parallel and accurate advantages.
This system have several characters
<ol>
    <li>Native framework, It imports linear algebra package, sparse matrix package and Delaunay algorithm only.</li>
    <li>Extremely concise, it is composed of 74 lines(non-comment) of codes, nearly magic.</li>
    <li>It can solve $n$-order linear PDEs, but this rely on the build-in elements and correct weak formulations are necessary.</li>
    <li>it supports you to use mix-elements and custom elements.</li>
</ol>
</font>
click http://www.li-zheng.net:8000/algorithms/simple_FEM.html for more details.<br>
click http://www.li-zheng.net:8000/algorithms/symbol_FEM.html to know how to solve nonlinear PDEs, the relevant project is in https://github.com/LizhengMathAi/symbol_FEM.

<h1><b>Demo</b></h1>
This demo will show the solution of following problem and its <a><img src="https://github.com/LizhengMathAi/symbol_FEM/blob/main/src/5.png" /></a> error
<a><img src="https://github.com/LizhengMathAi/simple_FEM/blob/main/src/6.png" /></a>
In 3-dimensional case, <a><img src="https://github.com/LizhengMathAi/simple_FEM/blob/main/src/7.png" /></a> and<br>
<a><img src="https://github.com/LizhengMathAi/simple_FEM/blob/main/src/8.png" /></a>
In 2-dimensional case,<br>
<a><img src="https://github.com/LizhengMathAi/simple_FEM/blob/main/src/9.png" /></a>
Some solutions
<img src="https://github.com/LizhengMathAi/simple_FEM/blob/main/src/2d.png" /><br>
<img src="https://github.com/LizhengMathAi/simple_FEM/blob/main/src/3d.png" />
