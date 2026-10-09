# Capstone brief

Transcribed from Lightmatter's email to the capstone group (screenshot Kenny shared, Oct 2026):

> What we're thinking is the following:
> - Develop an FPGA (or simulation) model of core photonics components such as MRMs --> Lightmatter can help the team
>   scope some of this out and provide guidance
> - The capstone team will then research on different locking and thermal control algos that could be developed (which
>   are novel/unique) to control the rings
> - Lightmatter + capstone team will sync to iterate on the algo and add on complexities and/or more photonic model
>   components as well
>
> The goal here is to have something your team can showcase to illustrate a new approach to stabilization for some core
> photonics components. We'd be open to doing this for interleavers or polarization controllers as well instead of (or
> in addition to) MRMs.

Kenny's requirements for this repo (Oct 2026):
- the most accurate MRM model possible, based on public Lightmatter information and the sources in his textbook
  (/Users/kenny/interactive-textbooks/lightmatter-capstone) and the OFC 2026 M4B.3 paper;
- test multiple types of control algorithms, and vary the model, through a single interface / set of parameters;
- an intuitive explanation of what was built, which references the textbook and disagrees with it where warranted;
- the most accurate tools that are not behind paywalls or licences; careful with RAM (16 GB machine).
