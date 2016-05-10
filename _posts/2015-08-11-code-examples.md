---
layout: post
title:  Code Examples for Scientific Computing
---


#Learn Scientific Programming by Examples:

As far as my experience, learning  a new programming/scripting language 
can be much faster by going through examples, particularly if you have already
done programming in another language and are familiar 
with the concepts of programming, such as
control structure, loops, data types and so on. 

The following is a collection of code examples, some I wrote while I was
learning a new language, some are developed as computational tools in my research, 
some of
them are designed by my supervisor Matt Choptuik and used in his popular
PHYS210 and PHYS410 computational physics courses in UBC. Lastly, there are few links that I came across on the web and
found them to be good examples to learn the features of the programming
language they are written in. 

Most of the examples are focused on scientific computing.
I will hopefully add more examples in C++,
Python, Bash scripting and GPU Accelerators (OpenACC).

* **C Programming**:

  - [Introductory Example Code in C](codes/C_hello.html)
  - [Input/Output from/to Files in C](codes/C_IO.html)
  - [Handling Arrays/Matrices (Pointers) in C](codes/C_Array_Pointer.html)
  - [Reading Parameters from Text File Using BBHUTIL](codes/readparam.html)
  - [Reading Input Interactively, Harvard's CS50
    Simple I/O Library[>]](codes/cs50.c.html)
  - [Matt's Vector C Library, Scientific Data Format (SDF) I/O Using LIBBBHUTIL](codes/Mattutils.html)
  - [My Implementation of Single Call SDF Read](codes/read_sdf.html)
  - [Example of Working with SDF, Simple Implementation of SDF Merger](codes/sdfwork.html)

* **Matlab**:

  - [Simple Operations in Matlab](codes/Matlab_Operation.html)
  - Matt's example of Matlab programming:
    [[1[>]]](codes/matlab1.html), [[2[>]]](codes/matlab2.html), [[3[>]]](codes/matlab4.html)
  - [Introductory Programming Structures in Matlab](codes/Matlab_Programming.html)
  - [Reading 2D Data From SDF/Binary into Matlab](codes/Matlab_read_sdf_bin.html)
  - [Simple 1D Interpolation in Matlab](codes/Matlab_interpolate.html)
  - [Linear Multivariable Regression in Matlab](codes/fitexample.html)
  - [Weighted Multivariable Regression Using Polynomials, Covariance and Correlation
    Functions](codes/nthpoly.html)
  - [Logistic Regression, Simple Classification in Matlab](codes/logistic.html)

* **Parallel Computing, MPI and PAMR**:

  - [Introductory MPI and CPU Communication Code](codes/mpi_hello.html)
  - [Simple 2D Wave Equation Parallel Code Using PAMR](codes/pamr.html) See
    documentation of PAMR [[here]](http://laplace.physics.ubc.ca/People/arman/files/PAMR_ref.pdf)
  - [Example of Reduction/Distribution Between CPUs, Mixed Parallel/Serial Calculation in PAMR](codes/Reduction_Distribution.html)
  - [Collection Operator to a Single Grid Structure in MPI](codes/collect_mpi.html)
  - LLNL MPI examples:
    [[1[>]]](https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_array.c) ,
[[2[>]]](https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_wave.c) ,
[[3[>]]](https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_heat2D.c) 

* **Maple**:

  - [Essential Tools/Operations in Maple](codes/Maple_Intro.html)
  - [Introductory Programming in Maple](codes/Maple_Programming_Intro.html)
  - [Matt's Maple Programming Notes[>]](codes/maple-programming-labs.html)
  - [Advanced Maple Programming, FD Finite Differencing
	   Toolkit[>]](http://rmanak.github.io/FD)
  - [Tutorials in Finite Difference Method Using FD Maple toolkit[>]](http://rmanak.github.io/FD/tutorials.html)
  - [1D wave equation with periodic boundary condition](codes/w1dpb.html), the driver
      [C code](codes/w1dpb_driver.html).
  - [2D wave equation, parallel-ready, with fixed boundary
       condition](codes/w2dfb_pamr.html), the driver [C code](codes/w2dpfb_driver.html).
  - [FD: Finite Difference Toolkit in Maple, User Manual
      [>]](codes/http://laplace.physics.ubc.ca/People/arman/FD_doc/)

* **R Programming**:
 - [Introductory Tools of R, "Hello R!"](codes/HelloR.html)
 - [Control Structure in R](codes/ControlR.html)
 - [IO Facilities in R](codes/IOR.html)
 - [Data Frames in R](codes/DataFrameR.html)
 - [Functions, Lexical Scoping Rules in R](codes/FunctionsR.html)
 - [Loop Operators in R](codes/LoopsR.html)

* **Perl Scripting**:

  - [Introductory Script in Perl](codes/perl-variables.html)
  - [Simple Math](codes/perl-math.html)
  - [Input/Output Facilities in Perl](codes/perl-I-O.html)
  - [Control Structures in Perl](codes/perl-control.html)
  - [Subroutines in Perl](codes/perl-subroutine.html)
  - [File/Directory Handling in Perl](codes/perl-files.html)
  - [Hash/Database in Perl](codes/perl-hash.html)
  - [Regular Expressions in Perl](codes/perl-regexp.html)


* **FORTRAN**:
   - [Matt's Fortran Example, PHYS410 Course](codes/p410all.html)
   - [Matt's Fortran Vector Manipulation: LIBVUTIL and Simple
     I/O in Fortran: LIBUTILIO](codes/MattFortranUtils.html)
   - [Fortran 77 Examples[>]](http://laplace.physics.ubc.ca/People/matt/Teaching/05Fall/PHYS410/Notes_fortran.html)

* **HTML/CSS/Markdown**:
  - [Markdown HTML Formatting
    Example[>]](http://laplace.phas.ubc.ca/People/arman/sg/format.html)
  - [A Simple web
    template[>]](http://laplace.physics.ubc.ca/People/arman/sg/Template.html)
  - [Try Markdown live online](https://stackedit.io/)[>]
  - [A Great Set of CSS Modules](http://purecss.io/)[>]


