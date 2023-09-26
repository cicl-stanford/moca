# Project name

## General points

- for folder and file names: 
	+ don't use white space in either folder or filenames, use an underscore "_" instead
	+ (almost always) use lower case only
- always use relative paths in your code
	+ for example, to save a figure from an R script inside the `code/R/` folder the path should be "../../figures/figure_name.pdf"
- keep your folder structure organized
	+ we recommend adhering to the folder structure in this repository 
	+ more complex projects may have additional folders such as `videos/`, `papers/`, ...
- note: some of the folders are empty except for a `.keep` file
	+ the `.keep` file is just there to make sure that github includes the otherwise empty folder 
	+ feel free to delete the `.keep` file once you've added another file to that folder
- each code subfolder has a readme file that should be updated with information about the code scripts 
- use github issues to keep track of any larger decisions that we make along the way 
- make sure to create a slack channel for each project, link up the github repository with the slack channel, and add the people working on the project to the github repo and slack channel 
- see our lab wiki for more help: https://github.com/cicl-stanford/wiki/wiki

## Repository structure 

```
├── code
│   ├── R
│   ├── bash
│   ├── experiments
│   └── python
├── data
├── figures
├── papers
├── presentation
└── writeup
```

### code 

Put all your code here. Use a separate folder for scripts based on the programming language. 

#### experiments 

The experiments folder is for the online (or in lab) experiments. Each experiment should be in its own folder. When you run another experiment, make sure to create a new folder (so that we always know what an experiment looked like when it was run). In readme file for the experiments folder, provide a brief summary of each experiment. Also note down any additional information that may not be saved within each experiment (e.g. how much the payment was for MTurk participants).

### data 

Put your raw data files here. Any data wrangling to that file should happen in your code scripts. 

### figures 

Save all your figures here. You may want to include additional subfolder here such as `plots/`, `diagrams/` etc. 

### papers 

Put research papers here that are relevant for your project. 

### presentation

Put your project presentation here (e.g. your keynote, powerpoint, google slides, or pdf file).

### writeup 

Put all your writing here. This folder structure is likely to expand for more complex projects. For example, you could add a subfolders like folders `journal/cognition/submission/`, `proceedings/cogsci/resubmission/` etc. 
