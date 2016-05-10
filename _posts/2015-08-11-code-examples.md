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

  - [Introductory Example Code in C](C_hello.html)
  - [Input/Output from/to Files in C](C_IO.html)
  - [Handling Arrays/Matrices (Pointers) in C](C_Array_Pointer.html)
  - [Reading Parameters from Text File Using BBHUTIL](readparam.html)
  - [Reading Input Interactively, Harvard's CS50
    Simple I/O Library[>]](cs50.c.html)
  - [Matt's Vector C Library, Scientific Data Format (SDF) I/O Using LIBBBHUTIL](Mattutils.html)
  - [My Implementation of Single Call SDF Read](read_sdf.html)
  - [Example of Working with SDF, Simple Implementation of SDF Merger](sdfwork.html)

* **Matlab**:

  - [Simple Operations in Matlab](Matlab_Operation.html)
  - Matt's example of Matlab programming:
    [[1[>]]](matlab1.html), [[2[>]]](matlab2.html), [[3[>]]](matlab4.html)
  - [Introductory Programming Structures in Matlab](Matlab_Programming.html)
  - [Reading 2D Data From SDF/Binary into Matlab](Matlab_read_sdf_bin.html)
  - [Simple 1D Interpolation in Matlab](Matlab_interpolate.html)
  - [Linear Multivariable Regression in Matlab](fitexample.html)
  - [Weighted Multivariable Regression Using Polynomials, Covariance and Correlation
    Functions](nthpoly.html)
  - [Logistic Regression, Simple Classification in Matlab](logistic.html)

* **Parallel Computing, MPI and PAMR**:

  - [Introductory MPI and CPU Communication Code](mpi_hello.html)
  - [Simple 2D Wave Equation Parallel Code Using PAMR](pamr.html) See
    documentation of PAMR [[here]](http://laplace.physics.ubc.ca/People/arman/files/PAMR_ref.pdf)
  - [Example of Reduction/Distribution Between CPUs, Mixed Parallel/Serial Calculation in PAMR](Reduction_Distribution.html)
  - [Collection Operator to a Single Grid Structure in MPI](collect_mpi.html)
  - LLNL MPI examples:
    [[1[>]]](https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_array.c) ,
[[2[>]]](https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_wave.c) ,
[[3[>]]](https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_heat2D.c) 

* **Maple**:

  - [Essential Tools/Operations in Maple](Maple_Intro.html)
  - [Introductory Programming in Maple](Maple_Programming_Intro.html)
  - [Matt's Maple Programming Notes[>]](maple-programming-labs.html)
  - [Advanced Maple Programming, FD Finite Differencing
	   Toolkit[>]](FD_doc/FD.mpl.html)
  - [Tutorials in Finite Difference Method Using FD Maple toolkit[>]](FD/tutorials.html)
  - [1D wave equation with periodic boundary condition](w1dpb.html), the driver
      [C code](w1dpb_driver.html).
  - [2D wave equation, parallel-ready, with fixed boundary
       condition](w2dfb_pamr.html), the driver [C code](w2dpfb_driver.html).
  - [FD: Finite Difference Toolkit in Maple, User Manual
      [>]](http://laplace.physics.ubc.ca/People/arman/FD_doc/)

* **R Programming**:
 - [Introductory Tools of R, "Hello R!"](HelloR.html)
 - [Control Structure in R](ControlR.html)
 - [IO Facilities in R](IOR.html)
 - [Data Frames in R](DataFrameR.html)
 - [Functions, Lexical Scoping Rules in R](FunctionsR.html)
 - [Loop Operators in R](LoopsR.html)

* **Perl Scripting**:

  - [Introductory Script in Perl](perl-variables.html)
  - [Simple Math](perl-math.html)
  - [Input/Output Facilities in Perl](perl-I-O.html)
  - [Control Structures in Perl](perl-control.html)
  - [Subroutines in Perl](perl-subroutine.html)
  - [File/Directory Handling in Perl](perl-files.html)
  - [Hash/Database in Perl](perl-hash.html)
  - [Regular Expressions in Perl](perl-regexp.html)


* **FORTRAN**:
   - [Matt's Fortran Example, PHYS410 Course](p410all.html)
   - [Matt's Fortran Vector Manipulation: LIBVUTIL and Simple
     I/O in Fortran: LIBUTILIO](MattFortranUtils.html)
   - [Fortran 77 Examples[>]](http://laplace.physics.ubc.ca/People/matt/Teaching/05Fall/PHYS410/Notes_fortran.html)

* **HTML/CSS/Markdown**:
  - [Markdown HTML Formatting
    Example[>]](http://laplace.phas.ubc.ca/People/arman/sg/format.html)
  - [A Simple web
    template[>]](http://laplace.physics.ubc.ca/People/arman/sg/Template.html)
  - [Try Markdown live online](https://stackedit.io/)[>]
  - [A Great Set of CSS Modules](http://purecss.io/)[>]


