# Modern accelerated programming
A tutorial at [eScience24](http://www.escience-conference.org/2024/tutorials).

## Content and schedule

- 09:00 - [Overview of modern HW lecture](lectures/escience24_acc_lecture_1_modern_hw.pdf)
- 09:30 - [Intro C/C++ pragma-based parallelism lecture](lectures/escience24_acc_lecture_2_intro_mp.pdf)
- 09:45 - [Intro to parallel programming hands-on](handson/session_1/README.md)
- 10:15 - [Beyond basics in parallelism lecture](lectures/escience24_acc_lecture_3_beyond_basics.pdf)
- 10:30 - Break
- 11:00 - [More complex parallel programming hands-on](handson/session_2/README.md)
- 11:20 - [Declarative parallelism in C++ lecture](lectures/escience24_acc_lecture_4_decl.pdf)
- 11:40 - [Declarative parallel programming and on-your-own hands-on](handson/session_3/README.md)
- 12:05 - [Standard libraries and acceleration lecture](lectures/escience24_acc_lecture_5_stdlib.pdf)
- 12:15 - [Accelerating FFT compute hands-on](handson/session_4/README.md)
- 12:30 - End

## Hands-on environment

The hands-on part is performend on the [National Research Platform](https://nationalresearchplatform.org),
using Kubernetes-based Nautilus.

To access Nautilus, participant must use the `kubectl` command.
Installation instructions are available at [https://kubernetes.io/docs/tasks/tools/](https://kubernetes.io/docs/tasks/tools/).

The needed configuration file can be found [here](https://drive.google.com/file/d/1BOy08Dar4ZZTUVi9EHe583-KVQ6HLpMG/view?usp=sharing).

Several GPU pods have been provisioned in Nautilus.  
Please claim one of the 5 slots at a pod in [this spereadsheet](https://docs.google.com/spreadsheets/d/17SroOWnkjnwWWWKkKVHKb1Br_u__9cNJnHco5NjfNKc/edit?usp=share_link). Use only the pod you signed up for, to avoid excessive contention.


