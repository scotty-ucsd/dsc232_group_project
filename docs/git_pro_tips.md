## maybe helpful git
### clone
- create local copy
```
git clone git@github.com:scotty-ucsd/dsc232_group_project.git
```

### list branches
- list all local and remote branches
```
git branch -a
```

### create your own branch and checkout
```
git checkout -b branch-name-example
```

### repo status
- see what files have been change since late commit

```
git status
```

## bread and butter
- make sure you are on your branch
- type type type, code code
- save file(ex: my_file.txt), then...

```
git add my_file.txt
git commit -m "added my_file.txt"
git push
```

-note: to add multiple files 

```
git add *
```

- then add your commit message

```
git commit -m "added more sweet files""
```

- push changes to remote

```
git push
```


