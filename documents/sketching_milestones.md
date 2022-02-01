### Sketching Milestones and deliverables for Fornax s3 subsets

- Explore state of the art for how to do subsets on HST, Spitzer, and GALEX
  - There is a separate team sketching out the science case now. How do we interact with them?
  - Do we try to do this with cutout services on prem?
- Provide some MVP for some very simple thing. 
  - Do we know whether the Science WG can use cutouts? Yes, small or big cutouts. 
  - Important for validating our answers.
  - (Brigitta is our liaison to the Science Use Case WG)
-Formulate strategies to improve the state-of-the-art
  - Describe many approaches.
  - Header files might make a lot of sense
  - Test these against 
    - time domain: gPhoton and TESS as well.
      - gPhoton is generalizable to HEASARC
    - spectral domain (e.g. SPHEREx, JWST, DKIST??)
    - ASDF needs to be considered.
- Downselect and then start prototyping some solutions identified above
- Benchmark these prototypes
- compare the performance, costs, and other pro/cons of each solution.
- Deliver a useable system to the Science WG

- A full system that everyone uses to get at subsets of all astronomy date in s3




From Geert:

### Question "How to obtain HST+Spitzer+Galex postage stamps for a specific galaxy in COSMOS?"

### Task 1: Define state-of-the-art
- *Deliverable: notebook(s) demonstrating how this task can best be achieved with current tools and services.*

### Task 2: Formulate strategies to improve the state-of-the-art.
- *Deliverable: markdown file listing candidate solutions with some pseudocode.*

### Task 3: Prototype solutions identified above
e.g.,
- Use multiple byte-range requests to locate and subset postage stamps on the fly.
- Use archive-provided "header files" to speed up 3a.
- Deploy state-of-the-art solution from Task 1 in the cloud via AWS Lambda.
- *Deliverable: Python files with prototype implementations (i.e. functions) and demo notebooks.*

### Task 4: Benchmark and analyze.
- *Deliverable: markdown file comparing the performance, costs, and other pro/cons of each solution.*

