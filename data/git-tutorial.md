
## Git Basics

If you can read only one chapter to get going with Git, this is it. This chapter covers every basic command you need to do the vast majority of the things you'll eventually spend your time doing with Git. By the end of the chapter, you should be able to configure and initialize a repository, begin and stop tracking files, and stage and commit changes. We'll also show you how to set up Git to ignore certain files and file patterns, how to undo mistakes quickly and easily, how to browse the history of your project and view changes between commits, and how to push and pull from remote repositories.

## Getting A Git Repository

You typically obtain a Git repository in one of two ways:
1. You can take a local directory that is currently not under version control, and turn it into a Git repository, or 2. You can *clone* an existing Git repository from elsewhere.

In either case, you end up with a Git repository on your local machine, ready for work.

## Initializing A Repository In An Existing Directory

If you have a project directory that is currently not under version control and you want to start controlling it with Git, you first need to go to that project's directory. If you've never done this, it looks a little different depending on which system you're running: for Linux:
$ cd /home/user/my_project for macOS:
$ cd /Users/user/my_project for Windows:
$ cd C:/Users/user/my_project and type:
$ git init This creates a new subdirectory named .git that contains all of your necessary repository files - a Git repository skeleton. At this point, nothing in your project is tracked yet. See Git Internals for 26 more information about exactly what files are contained in the .git directory you just created.

If you want to start version-controlling existing files (as opposed to an empty directory), you should probably begin tracking those files and do an initial commit. You can accomplish that with a few git add commands that specify the files you want to track, followed by a git commit:
$ git add *.c

![32_image_0.png](32_image_0.png) $ git add LICENSE $ git commit -m 'Initial project version' We'll go over what these commands do in just a minute. At this point, you have a Git repository with tracked files and an initial commit.

## Cloning An Existing Repository

If you want to get a copy of an existing Git repository - for example, a project you'd like to contribute to - the command you need is git clone. If you're familiar with other VCSs such as Subversion, you'll notice that the command is "clone" and not "checkout". This is an important distinction - instead of getting just a working copy, Git receives a full copy of nearly all data that the server has. Every version of every file for the history of the project is pulled down by default when you run git clone. In fact, if your server disk gets corrupted, you can often use nearly any of the clones on any client to set the server back to the state it was in when it was cloned (you may lose some server-side hooks and such, but all the versioned data would be there - see Getting Git on a Server for more details).

You clone a repository with git clone <url>. For example, if you want to clone the Git linkable library called libgit2, you can do so like this:
$ git clone https://github.com/libgit2/libgit2 That creates a directory named libgit2, initializes a .git directory inside it, pulls down all the data for that repository, and checks out a working copy of the latest version. If you go into the new libgit2 directory that was just created, you'll see the project files in there, ready to be worked on or used.

If you want to clone the repository into a directory named something other than libgit2, you can specify the new directory name as an additional argument:
$ git clone https://github.com/libgit2/libgit2 mylibgit That command does the same thing as the previous one, but the target directory is called mylibgit.

Git has a number of different transfer protocols you can use. The previous example uses the https:// protocol, but you may also see git:// or user@server:path/to/repo.git, which uses the SSH
transfer protocol. Getting Git on a Server will introduce all of the available options the server can set up to access your Git repository and the pros and cons of each.

# Recording Changes To The Repository

At this point, you should have a *bona fide* Git repository on your local machine, and a checkout or working copy of all of its files in front of you. Typically, you'll want to start making changes and committing snapshots of those changes into your repository each time the project reaches a state you want to record. Remember that each file in your working directory can be in one of two states: *tracked* or untracked. Tracked files are files that were in the last snapshot, as well as any newly staged files; they can be unmodified, modified, or staged. In short, tracked files are files that Git knows about. Untracked files are everything else - any files in your working directory that were not in your last snapshot and are not in your staging area. When you first clone a repository, all of your files will be tracked and unmodified because Git just checked them out and you haven't edited anything. As you edit files, Git sees them as modified, because you've changed them since your last commit. As you work, you selectively stage these modified files and then commit all those staged changes, and the cycle repeats.

![33_image_0.png](33_image_0.png)

Figure 8. The lifecycle of the status of your files

## Checking The Status Of Your Files

The main tool you use to determine which files are in which state is the git status command. If you run this command directly after a clone, you should see something like this:
$ git status On branch master Your branch is up-to-date with 'origin/master'. nothing to commit, working tree clean This means you have a clean working directory; in other words, none of your tracked files are modified. Git also doesn't see any untracked files, or they would be listed here. Finally, the command tells you which branch you're on and informs you that it has not diverged from the same branch on the server. For now, that branch is always master, which is the default; you won't worry about it here. Git Branching will go over branches and references in detail.

![34_image_0.png](34_image_0.png)

GitHub changed the default branch name from master to main in mid-2020, and other Git hosts followed suit. So you may find that the default branch name in some newly created repositories is main and not master. In addition, the default branch name can be changed (as you have seen in Your default branch name), so you may see a different name for the default branch.

However, Git itself still uses master as the default, so we will use it throughout the book.

Let's say you add a new file to your project, a simple README file. If the file didn't exist before, and you run git status, you see your untracked file like so:
$ echo 'My Project' > README $ git status On branch master Your branch is up-to-date with 'origin/master'. Untracked files: (use "git add <file>..." to include in what will be committed) README nothing added to commit but untracked files present (use "git add" to track)
You can see that your new README file is untracked, because it's under the "Untracked files" heading in your status output. Untracked basically means that Git sees a file you didn't have in the previous snapshot (commit), and which hasn't yet been staged; Git won't start including it in your commit snapshots until you explicitly tell it to do so. It does this so you don't accidentally begin including generated binary files or other files that you did not mean to include. You do want to start including README, so let's start tracking the file.

## Tracking New Files

In order to begin tracking a new file, you use the command git add. To begin tracking the README
file, you can run this:
$ git add README
If you run your status command again, you can see that your README file is now tracked and staged to be committed:
$ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git restore --staged <file>..." to unstage) new file: README
You can tell that it's staged because it's under the "Changes to be committed" heading. If you commit at this point, the version of the file at the time you ran git add is what will be in the subsequent historical snapshot. You may recall that when you ran git init earlier, you then ran git add <files> - that was to begin tracking files in your directory. The git add command takes a path name for either a file or a directory; if it's a directory, the command adds all the files in that directory recursively.

## Staging Modified Files

Let's change a file that was already tracked. If you change a previously tracked file called CONTRIBUTING.md and then run your git status command again, you get something that looks like this:
$ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage) new file: README Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) modified: CONTRIBUTING.md The CONTRIBUTING.md file appears under a section named "Changes not staged for commit" - which means that a file that is tracked has been modified in the working directory but not yet staged. To stage it, you run the git add command. git add is a multipurpose command - you use it to begin tracking new files, to stage files, and to do other things like marking merge-conflicted files as resolved. It may be helpful to think of it more as "add precisely this content to the next commit" rather than "add this file to the project". Let's run git add now to stage the CONTRIBUTING.md file, and then run git status again:
$ git add CONTRIBUTING.md $ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage) new file: README
Both files are staged and will go into your next commit. At this point, suppose you remember one little change that you want to make in CONTRIBUTING.md before you commit it. You open it again and make that change, and you're ready to commit. However, let's run git status one more time:
$ vim CONTRIBUTING.md $ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage) new file: README modified: CONTRIBUTING.md Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) modified: CONTRIBUTING.md What the heck? Now CONTRIBUTING.md is listed as both staged and unstaged. How is that possible? It turns out that Git stages a file exactly as it is when you run the git add command. If you commit now, the version of CONTRIBUTING.md as it was when you last ran the git add command is how it will go into the commit, not the version of the file as it looks in your working directory when you run git commit. If you modify a file after you run git add, you have to run git add again to stage the latest version of the file:
$ git add CONTRIBUTING.md
$ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage) new file: README modified: CONTRIBUTING.md

## Short Status

While the git status output is pretty comprehensive, it's also quite wordy. Git also has a short status flag so you can see your changes in a more compact way. If you run git status -s or git status --short you get a far more simplified output from the command:
$ git status -s M README MM Rakefile A lib/git.rb M lib/simplegit.rb ?? LICENSE.txt New files that aren't tracked have a ?? next to them, new files that have been added to the staging area have an A, modified files have an M and so on. There are two columns to the output - the lefthand column indicates the status of the staging area and the right-hand column indicates the status of the working tree. So for example in that output, the README file is modified in the working directory but not yet staged, while the lib/simplegit.rb file is modified and staged. The Rakefile was modified, staged and then modified again, so there are changes to it that are both staged and unstaged.

## Ignoring Files

Often, you'll have a class of files that you don't want Git to automatically add or even show you as being untracked. These are generally automatically generated files such as log files or files produced by your build system. In such cases, you can create a file listing patterns to match them named .gitignore. Here is an example .gitignore file:
$ cat .gitignore *.[oa] *~
The first line tells Git to ignore any files ending in ".o" or ".a" - object and archive files that may be the product of building your code. The second line tells Git to ignore all files whose names end with a tilde (~), which is used by many text editors such as Emacs to mark temporary files. You may also include a log, tmp, or pid directory; automatically generated documentation; and so on. Setting up a
.gitignore file for your new repository before you get going is generally a good idea so you don't accidentally commit files that you really don't want in your Git repository.

The rules for the patterns you can put in the .gitignore file are as follows:
- Blank lines or lines starting with \# are ignored.

- Standard glob patterns work, and will be applied recursively throughout the entire working tree.

- You can start patterns with a forward slash (/) to avoid recursivity.

- You can end patterns with a forward slash (/) to specify a directory.

- You can negate a pattern by starting it with an exclamation point (!).

Glob patterns are like simplified regular expressions that shells use. An asterisk (*) matches zero or more characters; [abc] matches any character inside the brackets (in this case a, b, or c); a question mark (?) matches a single character; and brackets enclosing characters separated by a hyphen ([09]) matches any character between them (in this case 0 through 9). You can also use two asterisks to match nested directories; a/**/z would match a/z, a/b/z, a/b/c/z, and so on.

32
\# ignore all .a files *.a \# but do track lib.a, even though you're ignoring .a files above !lib.a \# only ignore the TODO file in the current directory, not subdir/TODO /TODO \# ignore all files in any directory named build build/
\# ignore doc/notes.txt, but not doc/server/arch.txt doc/*.txt \# ignore all .pdf files in the doc/ directory and any of its subdirectories doc/**/*.pdf GitHub maintains a fairly comprehensive list of good .gitignore file examples for dozens of projects and languages at https://github.com/github/gitignore if you want a starting point for your project.

In the simple case, a repository might have a single .gitignore file in its root directory, which applies recursively to the entire repository. However, it is also possible to have additional .gitignore files in subdirectories. The rules in these nested .gitignore files apply only to the files under the directory where they are located. The Linux kernel source repository has 206 .gitignore files.

![38_image_0.png](38_image_0.png)

It is beyond the scope of this book to get into the details of multiple .gitignore files; see man gitignore for the details.

## Viewing Your Staged And Unstaged Changes

If the git status command is too vague for you - you want to know exactly what you changed, not just which files were changed - you can use the git diff command. We'll cover git diff in more detail later, but you'll probably use it most often to answer these two questions: What have you changed but not yet staged? And what have you staged that you are about to commit? Although git status answers those questions very generally by listing the file names, git diff shows you the exact lines added and removed - the patch, as it were.

Let's say you edit and stage the README file again and then edit the CONTRIBUTING.md file without staging it. If you run your git status command, you once again see something like this:
$ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage) modified: README Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) modified: CONTRIBUTING.md To see what you've changed but not yet staged, type git diff with no other arguments:
$ git diff diff --git a/CONTRIBUTING.md b/CONTRIBUTING.md index 8ebb991..643e24f 100644 --- a/CONTRIBUTING.md +++ b/CONTRIBUTING.md @@ -65,7 +65,8 @@ branch directly, things can get messy. Please include a nice description of your changes when you submit your PR; if we have to read the whole diff to figure out why you're contributing in the first place, you're less likely to get feedback and have your change -merged in. +merged in. Also, split your changes into comprehensive chunks if your patch is +longer than a dozen lines. If you are starting to work on a particular area, feel free to submit a PR that highlights your work in progress (and note in the PR title that it's That command compares what is in your working directory with what is in your staging area. The result tells you the changes you've made that you haven't yet staged.

If you want to see what you've staged that will go into your next commit, you can use git diff
--staged. This command compares your staged changes to your last commit:
$ git diff --staged diff --git a/README b/README new file mode 100644 index 0000000..03902a1 --- /dev/null +++ b/README @@ -0,0 +1 @@ +My Project It's important to note that git diff by itself doesn't show all changes made since your last commit - only changes that are still unstaged. If you've staged all of your changes, git diff will give you no output.

For another example, if you stage the CONTRIBUTING.md file and then edit it, you can use git diff to see the changes in the file that are staged and the changes that are unstaged. If our environment looks like this:
$ git add CONTRIBUTING.md

![40_image_0.png](40_image_0.png)

![40_image_1.png](40_image_1.png) $ echo '\# test line' >> CONTRIBUTING.md $ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage)

![40_image_2.png](40_image_2.png)

![40_image_3.png](40_image_3.png)

![40_image_4.png](40_image_4.png)

Changes not staged for commit:

![40_image_5.png](40_image_5.png)

![40_image_6.png](40_image_6.png) (use "git add <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) modified: CONTRIBUTING.md

![40_image_7.png](40_image_7.png)

Now you can use git diff to see what is still unstaged:
$ git diff diff --git a/CONTRIBUTING.md b/CONTRIBUTING.md index 643e24f..87f08c8 100644 --- a/CONTRIBUTING.md +++ b/CONTRIBUTING.md @@ -119,3 +119,4 @@ at the \#\# Starter Projects See our [projects list](https://github.com/libgit2/libgit2/blob/development/PROJECTS.md). +\# test line and git diff --cached to see what you've staged so far (--staged and --cached are synonyms):
$ git diff --cached diff --git a/CONTRIBUTING.md b/CONTRIBUTING.md index 8ebb991..643e24f 100644 --- a/CONTRIBUTING.md +++ b/CONTRIBUTING.md @@ -65,7 +65,8 @@ branch directly, things can get messy. Please include a nice description of your changes when you submit your PR; if we have to read the whole diff to figure out why you're contributing in the first place, you're less likely to get feedback and have your change -merged in. +merged in. Also, split your changes into comprehensive chunks if your patch is +longer than a dozen lines. If you are starting to work on a particular area, feel free to submit a PR that highlights your work in progress (and note in the PR title that it's Git Diff in an External Tool

![41_image_0.png](41_image_0.png)

We will continue to use the git diff command in various ways throughout the rest of the book. There is another way to look at these diffs if you prefer a graphical or external diff viewing program instead. If you run git difftool instead of git diff, you can view any of these diffs in software like emerge, vimdiff and many more
(including commercial products). Run git difftool --tool-help to see what is available on your system.

## Committing Your Changes

Now that your staging area is set up the way you want it, you can commit your changes. Remember that anything that is still unstaged - any files you have created or modified that you haven't run git add on since you edited them - won't go into this commit. They will stay as modified files on your disk. In this case, let's say that the last time you ran git status, you saw that everything was staged, so you're ready to commit your changes. The simplest way to commit is to type git commit:
$ git commit Doing so launches your editor of choice.

![41_image_1.png](41_image_1.png)

This is set by your shell's EDITOR environment variable - usually vim or emacs, although you can configure it with whatever you want using the git config --global core.editor command as you saw in Getting Started.

The editor displays the following text (this example is a Vim screen):
\# Please enter the commit message for your changes. Lines starting \# with '\#' will be ignored, and an empty message aborts the commit. \# On branch master \# Your branch is up-to-date with 'origin/master'. \# \# Changes to be committed: \# new file: README \# modified: CONTRIBUTING.md \# ~ ~ ~ ".git/COMMIT_EDITMSG" 9L, 283C
You can see that the default commit message contains the latest output of the git status command commented out and one empty line on top. You can remove these comments and type your commit message, or you can leave them there to help you remember what you're committing.

![42_image_0.png](42_image_0.png)

For an even more explicit reminder of what you've modified, you can pass the -v option to git commit. Doing so also puts the diff of your change in the editor so you can see exactly what changes you're committing.

When you exit the editor, Git creates your commit with that commit message (with the comments and diff stripped out).

Alternatively, you can type your commit message inline with the commit command by specifying it after a -m flag, like this:
$ git commit -m "Story 182: fix benchmarks for speed" [master 463dc4f] Story 182: fix benchmarks for speed 2 files changed, 2 insertions(+) create mode 100644 README
Now you've created your first commit! You can see that the commit has given you some output about itself: which branch you committed to (master), what SHA-1 checksum the commit has (463dc4f), how many files were changed, and statistics about lines added and removed in the commit. Remember that the commit records the snapshot you set up in your staging area. Anything you didn't stage is still sitting there modified; you can do another commit to add it to your history. Every time you perform a commit, you're recording a snapshot of your project that you can revert to or compare to later.

## Skipping The Staging Area

Although it can be amazingly useful for crafting commits exactly how you want them, the staging area is sometimes a bit more complex than you need in your workflow. If you want to skip the staging area, Git provides a simple shortcut. Adding the -a option to the git commit command makes Git automatically stage every file that is already tracked before doing the commit, letting you skip the git add part:
$ git status On branch master Your branch is up-to-date with 'origin/master'. Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) modified: CONTRIBUTING.md no changes added to commit (use "git add" and/or "git commit -a") $ git commit -a -m 'Add new benchmarks' [master 83e38c7] Add new benchmarks Notice how you don't have to run git add on the CONTRIBUTING.md file in this case before you commit.

That's because the -a flag includes all changed files. This is convenient, but be careful; sometimes this flag will cause you to include unwanted changes.

## Removing Files

To remove a file from Git, you have to remove it from your tracked files (more accurately, remove it from your staging area) and then commit. The git rm command does that, and also removes the file from your working directory so you don't see it as an untracked file the next time around. If you simply remove the file from your working directory, it shows up under the "Changes not staged for commit" (that is, *unstaged*) area of your git status output:
$ rm PROJECTS.md

![43_image_0.png](43_image_0.png) $ git status On branch master Your branch is up-to-date with 'origin/master'. Changes not staged for commit: (use "git add/rm <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) deleted: PROJECTS.md

![43_image_1.png](43_image_1.png) no changes added to commit (use "git add" and/or "git commit -a")
Then, if you run git rm, it stages the file's removal:
$ git rm PROJECTS.md rm 'PROJECTS.md' $ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage) deleted: PROJECTS.md The next time you commit, the file will be gone and no longer tracked. If you modified the file or had already added it to the staging area, you must force the removal with the -f option. This is a safety feature to prevent accidental removal of data that hasn't yet been recorded in a snapshot and that can't be recovered from Git. Another useful thing you may want to do is to keep the file in your working tree but remove it from your staging area. In other words, you may want to keep the file on your hard drive but not have Git track it anymore. This is particularly useful if you forgot to add something to your .gitignore 38 file and accidentally staged it, like a large log file or a bunch of .a compiled files. To do this, use the --cached option:
$ git rm --cached README
You can pass files, directories, and file-glob patterns to the git rm command. That means you can do things such as:
$ git rm log/\*.log Note the backslash (\) in front of the *. This is necessary because Git does its own filename expansion in addition to your shell's filename expansion. This command removes all files that have the .log extension in the log/ directory. Or, you can do something like this:
$ git rm \*~
This command removes all files whose names end with a ~.

## Moving Files

Unlike many other VCSs, Git doesn't explicitly track file movement. If you rename a file in Git, no metadata is stored in Git that tells it you renamed the file. However, Git is pretty smart about figuring that out after the fact - we'll deal with detecting file movement a bit later.

Thus it's a bit confusing that Git has a mv command. If you want to rename a file in Git, you can run something like:
$ git mv file_from file_to and it works fine. In fact, if you run something like this and look at the status, you'll see that Git considers it a renamed file:
$ git mv README.md README $ git status On branch master Your branch is up-to-date with 'origin/master'. Changes to be committed: (use "git reset HEAD <file>..." to unstage) renamed: README.md -> README
However, this is equivalent to running something like this:
$ mv README.md README $ git rm README.md $ git add README
Git figures out that it's a rename implicitly, so it doesn't matter if you rename a file that way or with the mv command. The only real difference is that git mv is one command instead of three - it's a convenience function. More importantly, you can use any tool you like to rename a file, and address the add/rm later, before you commit.

## Viewing The Commit History

After you have created several commits, or if you have cloned a repository with an existing commit history, you'll probably want to look back to see what has happened. The most basic and powerful tool to do this is the git log command.

These examples use a very simple project called "simplegit". To get the project, run:
$ git clone https://github.com/schacon/simplegit-progit When you run git log in this project, you should get output that looks something like this:
$ git log commit ca82a6dff817ec66f44342007202690a93763949 Author: Scott Chacon <schacon@gee-mail.com> Date: Mon Mar 17 21:52:11 2008 -0700 Change version number

![45_image_0.png](45_image_0.png)

![45_image_1.png](45_image_1.png) commit 085bb3bcb608e1e8451d4b2432f8ecbe6306e7e7

![45_image_2.png](45_image_2.png)

![45_image_3.png](45_image_3.png)

![45_image_4.png](45_image_4.png) Author: Scott Chacon <schacon@gee-mail.com> Date: Sat Mar 15 16:40:33 2008 -0700 Remove unnecessary test

![45_image_5.png](45_image_5.png)

![45_image_6.png](45_image_6.png) commit a11bef06a3f659402fe7563abf99ad00de2209e6

![45_image_7.png](45_image_7.png)

![45_image_8.png](45_image_8.png) Author: Scott Chacon <schacon@gee-mail.com> Date: Sat Mar 15 10:31:28 2008 -0700 Initial commit By default, with no arguments, git log lists the commits made in that repository in reverse chronological order; that is, the most recent commits show up first. As you can see, this command lists each commit with its SHA-1 checksum, the author's name and email, the date written, and the commit message.

A huge number and variety of options to the git log command are available to show you exactly what you're looking for. Here, we'll show you some of the most popular.

40 One of the more helpful options is -p or --patch, which shows the difference (the *patch* output) introduced in each commit. You can also limit the number of log entries displayed, such as using -2 to show only the last two entries.

$ git log -p -2 commit ca82a6dff817ec66f44342007202690a93763949 Author: Scott Chacon <schacon@gee-mail.com> Date: Mon Mar 17 21:52:11 2008 -0700 Change version number diff --git a/Rakefile b/Rakefile index a874b73..8f94139 100644 --- a/Rakefile +++ b/Rakefile @@ -5,7 +5,7 @@ require 'rake/gempackagetask' spec = Gem::Specification.new do |s| s.platform = Gem::Platform::RUBY s.name = "simplegit" - s.version = "0.1.0" + s.version = "0.1.1" s.author = "Scott Chacon" s.email = "schacon@gee-mail.com" s.summary = "A simple gem for using Git in Ruby code." commit 085bb3bcb608e1e8451d4b2432f8ecbe6306e7e7 Author: Scott Chacon <schacon@gee-mail.com> Date: Sat Mar 15 16:40:33 2008 -0700 Remove unnecessary test diff --git a/lib/simplegit.rb b/lib/simplegit.rb index a0a60ae..47c6340 100644 --- a/lib/simplegit.rb +++ b/lib/simplegit.rb @@ -18,8 +18,3 @@ class SimpleGit end end - -if $0 == __FILE__ - git = SimpleGit.new - puts git.show -end This option displays the same information but with a diff directly following each entry. This is very helpful for code review or to quickly browse what happened during a series of commits that a collaborator has added. You can also use a series of summarizing options with git log. For example, if you want to see some abbreviated stats for each commit, you can use the --stat option:
$ git log --stat commit ca82a6dff817ec66f44342007202690a93763949 Author: Scott Chacon <schacon@gee-mail.com> Date: Mon Mar 17 21:52:11 2008 -0700
  Change version number Rakefile | 2 +- 1 file changed, 1 insertion(+), 1 deletion(-) commit 085bb3bcb608e1e8451d4b2432f8ecbe6306e7e7 Author: Scott Chacon <schacon@gee-mail.com> Date: Sat Mar 15 16:40:33 2008 -0700 Remove unnecessary test lib/simplegit.rb | 5 ----- 1 file changed, 5 deletions(-) commit a11bef06a3f659402fe7563abf99ad00de2209e6 Author: Scott Chacon <schacon@gee-mail.com> Date: Sat Mar 15 10:31:28 2008 -0700 Initial commit README | 6 ++++++ Rakefile | 23 +++++++++++++++++++++++ lib/simplegit.rb | 25 +++++++++++++++++++++++++ 3 files changed, 54 insertions(+)
As you can see, the --stat option prints below each commit entry a list of modified files, how many files were changed, and how many lines in those files were added and removed. It also puts a summary of the information at the end.

Another really useful option is --pretty. This option changes the log output to formats other than the default. A few prebuilt option values are available for you to use. The oneline value for this option prints each commit on a single line, which is useful if you're looking at a lot of commits. In addition, the short, full, and fuller values show the output in roughly the same format but with less or more information, respectively:
$ git log --pretty=oneline ca82a6dff817ec66f44342007202690a93763949 Change version number 085bb3bcb608e1e8451d4b2432f8ecbe6306e7e7 Remove unnecessary test a11bef06a3f659402fe7563abf99ad00de2209e6 Initial commit The most interesting option value is format, which allows you to specify your own log output format. This is especially useful when you're generating output for machine parsing - because you specify the format explicitly, you know it won't change with updates to Git:
$ git log --pretty=format:"%h - %an, %ar : %s" ca82a6d - Scott Chacon, 6 years ago : Change version number 085bb3b - Scott Chacon, 6 years ago : Remove unnecessary test a11bef0 - Scott Chacon, 6 years ago : Initial commit Useful specifiers for git log --pretty=format lists some of the more useful specifiers that format takes.

| Table 1. Useful specifiers for git log --pretty=format Specifier Description of Output %H Commit hash %h Abbreviated commit hash %T Tree hash %t Abbreviated tree hash %P Parent hashes %p Abbreviated parent hashes %an Author name %ae Author email %ad Author date (format respects the --date=option) %ar Author date, relative %cn Committer name %ce Committer email %cd Committer date %cr Committer date, relative %s Subject   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

You may be wondering what the difference is between *author* and *committer*. The author is the person who originally wrote the work, whereas the committer is the person who last applied the work. So, if you send in a patch to a project and one of the core members applies the patch, both of you get credit - you as the author, and the core member as the committer. We'll cover this distinction a bit more in Distributed Git.

The oneline and format option values are particularly useful with another log option called --graph.

This option adds a nice little ASCII graph showing your branch and merge history:
$ git log --pretty=format:"%h %s" --graph * 2d3acf9 Ignore errors from SIGCHLD on trap * 5e3ee11 Merge branch 'master' of https://github.com/dustin/grit.git |\ | * 420eac9 Add method for getting the current branch * | 30e367c Timeout code and tests * | 5a09431 Add timeout protection to grit * | e1193f8 Support for heads with slashes in them |/ * d6016bc Require time for xmlschema * 11d191e Merge branch 'defunkt' into local This type of output will become more interesting as we go through branching and merging in the next chapter.

Those are only some simple output-formatting options to git log - there are many more. Common options to git log lists the options we've covered so far, as well as some other common formatting options that may be useful, along with how they change the output of the log command.

| Table 2. Common options to git log Option Description -p Show the patch introduced with each commit. --stat Show statistics for files modified in each commit. --shortstat Display only the changed/insertions/deletions line from the --stat command. --name-only Show the list of files modified after the commit information. --name-status Show the list of files affected with added/modified/deleted information as well. --abbrev-commit Show only the first few characters of the SHA-1 checksum instead of all 40. --relative-date Display the date in a relative format (for example, "2 weeks ago") instead of using the full date format. --graph Display an ASCII graph of the branch and merge history beside the log output. --pretty Show commits in an alternate format. Option values include oneline, short, full, fuller, and format (where you specify your own format). --oneline Shorthand for --pretty=oneline --abbrev-commit used together.   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Limiting Log Output

In addition to output-formatting options, git log takes a number of useful limiting options; that is, options that let you show only a subset of commits. You've seen one such option already - the -2 option, which displays only the last two commits. In fact, you can do -<n>, where n is any integer to show the last n commits. In reality, you're unlikely to use that often, because Git by default pipes all output through a pager so you see only one page of log output at a time.

However, the time-limiting options such as --since and --until are very useful. For example, this command gets the list of commits made in the last two weeks:
$ git log --since=2.weeks This command works with lots of formats - you can specify a specific date like "2008-01-15", or a relative date such as "2 years 1 day 3 minutes ago".

44 You can also filter the list to commits that match some search criteria. The --author option allows you to filter on a specific author, and the --grep option lets you search for keywords in the commit messages.

![50_image_0.png](50_image_0.png)

You can specify more than one instance of both the --author and --grep search criteria, which will limit the commit output to commits that match any of the
--author patterns and any of the --grep patterns; however, adding the --all-match option further limits the output to just those commits that match all --grep patterns.

Another really helpful filter is the -S option (colloquially referred to as Git's "pickaxe" option),
which takes a string and shows only those commits that changed the number of occurrences of that string. For instance, if you wanted to find the last commit that added or removed a reference to a specific function, you could call:
$ git log -S function_name The last really useful option to pass to git log as a filter is a path. If you specify a directory or file name, you can limit the log output to commits that introduced a change to those files. This is always the last option and is generally preceded by double dashes (--) to separate the paths from the options:
$ git log -- path/to/file In Options to limit the output of git log we'll list these and a few other common options for your reference.

| Table 3. Options to limit the output of git log Option Description -<n> Show only the last n commits. --since, --after Limit the commits to those made after the specified date. --until, --before Limit the commits to those made before the specified date. --author Only show commits in which the author entry matches the specified string. --committer Only show commits in which the committer entry matches the specified string. --grep Only show commits with a commit message containing the string. -S Only show commits adding or removing code matching the string.   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

For example, if you want to see which commits modifying test files in the Git source code history were committed by Junio Hamano in the month of October 2008 and are not merge commits, you
$ git log --pretty="%h - %s" --author='Junio C Hamano' --since="2008-10-01" \ --before="2008-11-01" --no-merges -- t/ 5610e3b - Fix testcase failure when extended attributes are in use acd3b9e - Enhance hold_lock_file_for_{update,append}() API f563754 - demonstrate breakage of detached checkout with symbolic link HEAD d1a43f2 - reset --hard/read-tree --reset -u: remove unmerged new paths 51a94af - Fix "checkout --track -b newbranch" on detached HEAD b0ad11e - pull: allow "git pull origin $something:$current_branch" into an unborn branch Of the nearly 40,000 commits in the Git source code history, this command shows the 6 that match those criteria.

Preventing the display of merge commits

![51_image_0.png](51_image_0.png)

Depending on the workflow used in your repository, it's possible that a sizable percentage of the commits in your log history are just merge commits, which typically aren't very informative. To prevent the display of merge commits cluttering up your log history, simply add the log option --no-merges.

## Undoing Things

At any stage, you may want to undo something. Here, we'll review a few basic tools for undoing changes that you've made. Be careful, because you can't always undo some of these undos. This is one of the few areas in Git where you may lose some work if you do it wrong. One of the common undos takes place when you commit too early and possibly forget to add some files, or you mess up your commit message. If you want to redo that commit, make the additional changes you forgot, stage them, and commit again using the --amend option:
$ git commit --amend This command takes your staging area and uses it for the commit. If you've made no changes since your last commit (for instance, you run this command immediately after your previous commit), then your snapshot will look exactly the same, and all you'll change is your commit message. The same commit-message editor fires up, but it already contains the message of your previous commit. You can edit the message the same as always, but it overwrites your previous commit. As an example, if you commit and then realize you forgot to stage the changes in a file you wanted to add to this commit, you can do something like this:
$ git commit -m 'Initial commit' $ git add forgotten_file You end up with a single commit - the second commit replaces the results of the first.

It's important to understand that when you're amending your last commit, you're not so much fixing it as *replacing* it entirely with a new, improved commit that pushes the old commit out of the way and puts the new commit in its place. Effectively, it's as if the previous commit never happened, and it won't show up in your repository history.

![52_image_0.png](52_image_0.png)

The obvious value to amending commits is to make minor improvements to your last commit, without cluttering your repository history with commit messages of the form, "Oops, forgot to add a file" or "Darn, fixing a typo in last commit".

![52_image_1.png](52_image_1.png)

Only amend commits that are still local and have not been pushed somewhere. Amending previously pushed commits and force pushing the branch will cause problems for your collaborators. For more on what happens when you do this and how to recover if you're on the receiving end read The Perils of Rebasing.

## Unstaging A Staged File

The next two sections demonstrate how to work with your staging area and working directory changes. The nice part is that the command you use to determine the state of those two areas also reminds you how to undo changes to them. For example, let's say you've changed two files and want to commit them as two separate changes, but you accidentally type git add * and stage them both. How can you unstage one of the two? The git status command reminds you:
$ git add * $ git status On branch master Changes to be committed: (use "git reset HEAD <file>..." to unstage) renamed: README.md -> README modified: CONTRIBUTING.md Right below the "Changes to be committed" text, it says use git reset HEAD <file>… to unstage. So, let's use that advice to unstage the CONTRIBUTING.md file:
$ git reset HEAD CONTRIBUTING.md Unstaged changes after reset: M CONTRIBUTING.md $ git status On branch master Changes to be committed: (use "git reset HEAD <file>..." to unstage) Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) modified: CONTRIBUTING.md The command is a bit strange, but it works. The CONTRIBUTING.md file is modified but once again unstaged.

![53_image_0.png](53_image_0.png)

It's true that git reset can be a dangerous command, especially if you provide the
--hard flag. However, in the scenario described above, the file in your working directory is not touched, so it's relatively safe.

For now this magic invocation is all you need to know about the git reset command. We'll go into much more detail about what reset does and how to master it to do really interesting things in Reset Demystified.

## Unmodifying A Modified File

What if you realize that you don't want to keep your changes to the CONTRIBUTING.md file? How can you easily unmodify it - revert it back to what it looked like when you last committed (or initially cloned, or however you got it into your working directory)? Luckily, git status tells you how to do that, too. In the last example output, the unstaged area looks like this:
Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git checkout -- <file>..." to discard changes in working directory) modified: CONTRIBUTING.md It tells you pretty explicitly how to discard the changes you've made. Let's do what it says:
$ git checkout -- CONTRIBUTING.md $ git status On branch master Changes to be committed: (use "git reset HEAD <file>..." to unstage) renamed: README.md -> README
You can see that the changes have been reverted.

![53_image_1.png](53_image_1.png)

It's important to understand that git checkout -- <file> is a dangerous command.

Any local changes you made to that file are gone - Git just replaced that file with the last staged or committed version. Don't ever use this command unless you 48 If you would like to keep the changes you've made to that file but still need to get it out of the way for now, we'll go over stashing and branching in Git Branching; these are generally better ways to go. Remember, anything that is *committed* in Git can almost always be recovered. Even commits that were on branches that were deleted or commits that were overwritten with an --amend commit can be recovered (see Data Recovery for data recovery). However, anything you lose that was never committed is likely never to be seen again.

## Undoing Things With Git Restore

Git version 2.23.0 introduced a new command: git restore. It's basically an alternative to git reset which we just covered. From Git version 2.23.0 onwards, Git will use git restore instead of git reset for many undo operations.

Let's retrace our steps, and undo things with git restore instead of git reset.

## Unstaging A Staged File With Git Restore

The next two sections demonstrate how to work with your staging area and working directory changes with git restore. The nice part is that the command you use to determine the state of those two areas also reminds you how to undo changes to them. For example, let's say you've changed two files and want to commit them as two separate changes, but you accidentally type git add * and stage them both. How can you unstage one of the two? The git status command reminds you:
$ git add * $ git status On branch master Changes to be committed: (use "git restore --staged <file>..." to unstage) modified: CONTRIBUTING.md renamed: README.md -> README
Right below the "Changes to be committed" text, it says use git restore --staged <file>… to unstage. So, let's use that advice to unstage the CONTRIBUTING.md file:
$ git restore --staged CONTRIBUTING.md $ git status On branch master Changes to be committed: (use "git restore --staged <file>..." to unstage) renamed: README.md -> README Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git restore <file>..." to discard changes in working directory)
The CONTRIBUTING.md file is modified but once again unstaged.

## Unmodifying A Modified File With Git Restore

What if you realize that you don't want to keep your changes to the CONTRIBUTING.md file? How can you easily unmodify it - revert it back to what it looked like when you last committed (or initially cloned, or however you got it into your working directory)? Luckily, git status tells you how to do that, too. In the last example output, the unstaged area looks like this:
Changes not staged for commit: (use "git add <file>..." to update what will be committed) (use "git restore <file>..." to discard changes in working directory) modified: CONTRIBUTING.md It tells you pretty explicitly how to discard the changes you've made. Let's do what it says:
$ git restore CONTRIBUTING.md $ git status On branch master Changes to be committed: (use "git restore --staged <file>..." to unstage) renamed: README.md -> README

![55_image_0.png](55_image_0.png)

It's important to understand that git restore <file> is a dangerous command. Any local changes you made to that file are gone - Git just replaced that file with the last staged or committed version. Don't ever use this command unless you absolutely know that you don't want those unsaved local changes.

## Working With Remotes

To be able to collaborate on any Git project, you need to know how to manage your remote repositories. Remote repositories are versions of your project that are hosted on the Internet or network somewhere. You can have several of them, each of which generally is either read-only or read/write for you. Collaborating with others involves managing these remote repositories and pushing and pulling data to and from them when you need to share work. Managing remote repositories includes knowing how to add remote repositories, remove remotes that are no longer valid, manage various remote branches and define them as being tracked or not, and more. In this section, we'll cover some of these remote-management skills.

Remote repositories can be on your local machine.

![55_image_1.png](55_image_1.png)

It is entirely possible that you can be working with a "remote" repository that is, in fact, on the same host you are. The word "remote" does not necessarily imply that the repository is somewhere else on the network or Internet, only that it is elsewhere. Working with such a remote repository would still involve all the standard pushing, pulling and fetching operations as with any other remote.

## Showing Your Remotes

To see which remote servers you have configured, you can run the git remote command. It lists the shortnames of each remote handle you've specified. If you've cloned your repository, you should at least see origin - that is the default name Git gives to the server you cloned from:
$ git clone https://github.com/schacon/ticgit Cloning into 'ticgit'... remote: Reusing existing pack: 1857, done. remote: Total 1857 (delta 0), reused 0 (delta 0)
Receiving objects: 100% (1857/1857), 374.35 KiB | 268.00 KiB/s, done.

Resolving deltas: 100% (772/772), done. Checking connectivity... done. $ cd ticgit $ git remote origin You can also specify -v, which shows you the URLs that Git has stored for the shortname to be used when reading and writing to that remote:
$ git remote -v origin https://github.com/schacon/ticgit (fetch) origin https://github.com/schacon/ticgit (push)
If you have more than one remote, the command lists them all. For example, a repository with multiple remotes for working with several collaborators might look something like this.

$ cd grit

![56_image_0.png](56_image_0.png)

![56_image_1.png](56_image_1.png) $ git remote -v bakkdoor https://github.com/bakkdoor/grit (fetch) bakkdoor https://github.com/bakkdoor/grit (push) cho45 https://github.com/cho45/grit (fetch) cho45 https://github.com/cho45/grit (push) defunkt https://github.com/defunkt/grit (fetch) defunkt https://github.com/defunkt/grit (push) koke git://github.com/koke/grit.git (fetch) koke git://github.com/koke/grit.git (push) origin git@github.com:mojombo/grit.git (fetch) origin git@github.com:mojombo/grit.git (push)
This means we can pull contributions from any of these users pretty easily. We may additionally have permission to push to one or more of these, though we can't tell that here. Notice that these remotes use a variety of protocols; we'll cover more about this in Getting Git on a

## Adding Remote Repositories

We've mentioned and given some demonstrations of how the git clone command implicitly adds the origin remote for you. Here's how to add a new remote explicitly. To add a new remote Git repository as a shortname you can reference easily, run git remote add <shortname> <url>:
$ git remote origin $ git remote add pb https://github.com/paulboone/ticgit $ git remote -v origin https://github.com/schacon/ticgit (fetch) origin https://github.com/schacon/ticgit (push) pb https://github.com/paulboone/ticgit (fetch) pb https://github.com/paulboone/ticgit (push)
Now you can use the string pb on the command line instead of the whole URL. For example, if you want to fetch all the information that Paul has but that you don't yet have in your repository, you can run git fetch pb:
$ git fetch pb

![57_image_0.png](57_image_0.png) remote: Counting objects: 43, done. remote: Compressing objects: 100% (36/36), done. remote: Total 43 (delta 10), reused 31 (delta 5) Unpacking objects: 100% (43/43), done. From https://github.com/paulboone/ticgit * [new branch] master -> pb/master * [new branch] ticgit -> pb/ticgit Paul's master branch is now accessible locally as pb/master - you can merge it into one of your

![57_image_1.png](57_image_1.png)

branches, or you can check out a local branch at that point if you want to inspect it. We'll go over what branches are and how to use them in much more detail in Git Branching.

## Fetching And Pulling From Your Remotes

As you just saw, to get data from your remote projects, you can run:
$ git fetch <remote>
The command goes out to that remote project and pulls down all the data from that remote project that you don't have yet. After you do this, you should have references to all the branches from that remote, which you can merge in or inspect at any time. If you clone a repository, the command automatically adds that remote repository under the name
"origin". So, git fetch origin fetches any new work that has been pushed to that server since you cloned (or last fetched from) it. It's important to note that the git fetch command only downloads the data to your local repository - it doesn't automatically merge it with any of your work or modify what you're currently working on. You have to merge it manually into your work when you're ready. If your current branch is set up to track a remote branch (see the next section and Git Branching for more information), you can use the git pull command to automatically fetch and then merge that remote branch into your current branch. This may be an easier or more comfortable workflow for you; and by default, the git clone command automatically sets up your local master branch to track the remote master branch (or whatever the default branch is called) on the server you cloned from.

Running git pull generally fetches data from the server you originally cloned from and automatically tries to merge it into the code you're currently working on.

From Git version 2.27 onward, git pull will give a warning if the pull.rebase variable is not set. Git will keep warning you until you set the variable.

![58_image_0.png](58_image_0.png)

If you want the default behavior of Git (fast-forward if possible, else create a merge commit): git config --global pull.rebase "false" If you want to rebase when pulling: git config --global pull.rebase "true"

## Pushing To Your Remotes

When you have your project at a point that you want to share, you have to push it upstream. The command for this is simple: git push <remote> <branch>. If you want to push your master branch to your origin server (again, cloning generally sets up both of those names for you automatically),
then you can run this to push any commits you've done back up to the server:
$ git push origin master This command works only if you cloned from a server to which you have write access and if nobody has pushed in the meantime. If you and someone else clone at the same time and they push upstream and then you push upstream, your push will rightly be rejected. You'll have to fetch their work first and incorporate it into yours before you'll be allowed to push. See Git Branching for more detailed information on how to push to remote servers.

## Inspecting A Remote

If you want to see more information about a particular remote, you can use the git remote show
<remote> command. If you run this command with a particular shortname, such as origin, you get something like this:
$ git remote show origin * remote origin Fetch URL: https://github.com/schacon/ticgit Push URL: https://github.com/schacon/ticgit HEAD branch: master Remote branches: master tracked dev-branch tracked Local branch configured for 'git pull': master merges with remote master Local ref configured for 'git push': master pushes to master (up to date)
It lists the URL for the remote repository as well as the tracking branch information. The command helpfully tells you that if you're on the master branch and you run git pull, it will automatically merge the remote's master branch into the local one after it has been fetched. It also lists all the remote references it has pulled down. That is a simple example you're likely to encounter. When you're using Git more heavily, however, you may see much more information from git remote show:

![59_image_0.png](59_image_0.png)

This command shows which branch is automatically pushed to when you run git push while on

| Remote branches:   master                                                                                                                                                                         | tracked                      |                                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|-----------------------------------------------|
|                                                                                                                                                                                                   | dev-branch                   | tracked                                       |
|                                                                                                                                                                                                   | markdown-strip               | tracked                                       |
|                                                                                                                                                                                                   | issue-43                     | new (next fetch will store in remotes/origin) |
|                                                                                                                                                                                                   | issue-45                     | new (next fetch will store in remotes/origin) |
|                                                                                                                                                                                                   | refs/remotes/origin/issue-11 | stale (use 'git remote prune' to remove)      |
| Local branches configured for 'git pull':   dev-branch merges with remote dev-branch   master merges with remote master   Local refs configured for 'git push':   dev-branch pushes to dev-branch | (up to                       |                                               |
| date)   markdown-strip                                                                                                                                                                            | pushes to markdown-strip     | (up to                                        |
| date)   master                                                                                                                                                                                    | pushes to master             | (up to                                        |
| date)                                                                                                                                                                                             |                              |                                               |

certain branches. It also shows you which remote branches on the server you don't yet have, which remote branches you have that have been removed from the server, and multiple local branches that are able to merge automatically with their remote-tracking branch when you run git pull.

## Renaming And Removing Remotes

You can run git remote rename to change a remote's shortname. For instance, if you want to rename pb to paul, you can do so with git remote rename:
54
$ git remote rename pb paul $ git remote origin paul It's worth mentioning that this changes all your remote-tracking branch names, too. What used to be referenced at pb/master is now at paul/master.

If you want to remove a remote for some reason - you've moved the server or are no longer using a particular mirror, or perhaps a contributor isn't contributing anymore - you can either use git remote remove or git remote rm:
$ git remote remove paul $ git remote origin Once you delete the reference to a remote this way, all remote-tracking branches and configuration settings associated with that remote are also deleted.

## Tagging

Like most VCSs, Git has the ability to tag specific points in a repository's history as being important.

Typically, people use this functionality to mark release points (v1.0, v2.0 and so on). In this section, you'll learn how to list existing tags, how to create and delete tags, and what the different types of tags are.

## Listing Your Tags

Listing the existing tags in Git is straightforward. Just type git tag (with optional -l or --list):
$ git tag v1.0 v2.0 This command lists the tags in alphabetical order; the order in which they are displayed has no real importance. You can also search for tags that match a particular pattern. The Git source repo, for instance, contains more than 500 tags. If you're interested only in looking at the 1.8.5 series, you can run this:
$ git tag -l "v1.8.5*" v1.8.5 v1.8.5-rc0 v1.8.5-rc1 v1.8.5-rc2 v1.8.5-rc3 v1.8.5.1 v1.8.5.2 v1.8.5.3 v1.8.5.4 v1.8.5.5 Listing tag wildcards requires -l *or* --list *option*

![61_image_0.png](61_image_0.png)

If you want just the entire list of tags, running the command git tag implicitly assumes you want a listing and provides one; the use of -l or --list in this case is optional.

If, however, you're supplying a wildcard pattern to match tag names, the use of -l or --list is mandatory.

## Creating Tags

Git supports two types of tags: *lightweight* and *annotated*. A lightweight tag is very much like a branch that doesn't change - it's just a pointer to a specific commit. Annotated tags, however, are stored as full objects in the Git database. They're checksummed; contain the tagger name, email, and date; have a tagging message; and can be signed and verified with GNU Privacy Guard (GPG). It's generally recommended that you create annotated tags so you can have all this information; but if you want a temporary tag or for some reason don't want to keep the other information, lightweight tags are available too.

## Annotated Tags

Creating an annotated tag in Git is simple. The easiest way is to specify -a when you run the tag command:
$ git tag -a v1.4 -m "my version 1.4" $ git tag v0.1 v1.3 v1.4 The -m specifies a tagging message, which is stored with the tag. If you don't specify a message for an annotated tag, Git launches your editor so you can type it in.

You can see the tag data along with the commit that was tagged by using the git show command:
$ git show v1.4 tag v1.4 Tagger: Ben Straub <ben@straub.cc> Date: Sat May 3 20:19:12 2014 -0700 56 my version 1.4 commit ca82a6dff817ec66f44342007202690a93763949 Author: Scott Chacon <schacon@gee-mail.com> Date: Mon Mar 17 21:52:11 2008 -0700 Change version number That shows the tagger information, the date the commit was tagged, and the annotation message before showing the commit information.

## Lightweight Tags

Another way to tag commits is with a lightweight tag. This is basically the commit checksum stored in a file - no other information is kept. To create a lightweight tag, don't supply any of the -a, -s, or
-m options, just provide a tag name:
$ git tag v1.4-lw $ git tag v0.1 v1.3 v1.4 v1.4-lw v1.5 This time, if you run git show on the tag, you don't see the extra tag information. The command just shows the commit:
$ git show v1.4-lw commit ca82a6dff817ec66f44342007202690a93763949 Author: Scott Chacon <schacon@gee-mail.com> Date: Mon Mar 17 21:52:11 2008 -0700 Change version number

## Tagging Later

You can also tag commits after you've moved past them. Suppose your commit history looks like this:
$ git log --pretty=oneline 15027957951b64cf874c3557a0f3547bd83b3ff6 Merge branch 'experiment' a6b4c97498bd301d84096da251c98a07c7723e65 Create write support 0d52aaab4479697da7686c15f77a3d64d9165190 One more thing 6d52a271eda8725415634dd79daabbc4d9b6008e Merge branch 'experiment' 0b7434d86859cc7b8c3d5e1dddfed66ff742fcbc Add commit function 4682c3261057305bdd616e23b64b0857d832627b Add todo file 166ae0c4d3f420721acbb115cc33848dfcc2121a Create write support 9fceb02d0ae598e95dc970b74767f19372d61af8 Update rakefile 964f16d36dfccde844893cac5b347e7b3d44abbc Commit the todo 8a5cbc430f1a9c3d00faaeffd07798508422908a Update readme Now, suppose you forgot to tag the project at v1.2, which was at the "Update rakefile" commit. You can add it after the fact. To tag that commit, you specify the commit checksum (or part of it) at the end of the command:
$ git tag -a v1.2 9fceb02 You can see that you've tagged the commit:
$ git tag v0.1 v1.2 v1.3 v1.4 v1.4-lw v1.5 $ git show v1.2 tag v1.2 Tagger: Scott Chacon <schacon@gee-mail.com> Date: Mon Feb 9 15:32:16 2009 -0800 version 1.2 commit 9fceb02d0ae598e95dc970b74767f19372d61af8 Author: Magnus Chacon <mchacon@gee-mail.com> Date: Sun Apr 27 20:43:35 2008 -0700 Update rakefile ...

## Sharing Tags

By default, the git push command doesn't transfer tags to remote servers. You will have to explicitly push tags to a shared server after you have created them. This process is just like sharing remote branches - you can run git push origin <tagname>.

$ git push origin v1.5 Counting objects: 14, done. Delta compression using up to 8 threads. Compressing objects: 100% (12/12), done. Writing objects: 100% (14/14), 2.05 KiB | 0 bytes/s, done. Total 14 (delta 3), reused 0 (delta 0) To git@github.com:schacon/simplegit.git * [new tag] v1.5 -> v1.5 If you have a lot of tags that you want to push up at once, you can also use the --tags option to the git push command. This will transfer all of your tags to the remote server that are not already there.

$ git push origin --tags Counting objects: 1, done. Writing objects: 100% (1/1), 160 bytes | 0 bytes/s, done. Total 1 (delta 0), reused 0 (delta 0) To git@github.com:schacon/simplegit.git * [new tag] v1.4 -> v1.4 * [new tag] v1.4-lw -> v1.4-lw Now, when someone else clones or pulls from your repository, they will get all your tags as well.

git push *pushes both types of tags*

![64_image_0.png](64_image_0.png)

git push <remote> --tags will push both lightweight and annotated tags. There is currently no option to push only lightweight tags, but if you use git push <remote>
--follow-tags only annotated tags will be pushed to the remote.

## Deleting Tags

To delete a tag on your local repository, you can use git tag -d <tagname>. For example, we could remove our lightweight tag above as follows:
$ git tag -d v1.4-lw Deleted tag 'v1.4-lw' (was e7d5add)
Note that this does not remove the tag from any remote servers. There are two common variations for deleting a tag from a remote server.

The first variation is git push <remote> :refs/tags/<tagname>:
$ git push origin :refs/tags/v1.4-lw To /git@github.com:schacon/simplegit.git - [deleted] v1.4-lw The way to interpret the above is to read it as the null value before the colon is being pushed to the remote tag name, effectively deleting it. The second (and more intuitive) way to delete a remote tag is with:
$ git push origin --delete <tagname>

## Checking Out Tags

If you want to view the versions of files a tag is pointing to, you can do a git checkout of that tag, although this puts your repository in "detached HEAD" state, which has some ill side effects:
$ git checkout v2.0.0 Note: switching to 'v2.0.0'. You are in 'detached HEAD' state. You can look around, make experimental changes and commit them, and you can discard any commits you make in this state without impacting any branches by performing another checkout. If you want to create a new branch to retain commits you create, you may do so (now or later) by using -c with the switch command. Example: git switch -c <new-branch-name> Or undo this operation with: git switch - Turn off this advice by setting config variable advice.detachedHead to false HEAD is now at 99ada87... Merge pull request \#89 from schacon/appendix-final $ git checkout v2.0-beta-0.1 Previous HEAD position was 99ada87... Merge pull request \#89 from schacon/appendixfinal HEAD is now at df3f601... Add atlas.json and cover image In "detached HEAD" state, if you make changes and then create a commit, the tag will stay the same, but your new commit won't belong to any branch and will be unreachable, except by the exact commit hash. Thus, if you need to make changes - say you're fixing a bug on an older version, for instance - you will generally want to create a branch:
$ git checkout -b version2 v2.0.0 Switched to a new branch 'version2' If you do this and make a commit, your version2 branch will be slightly different than your v2.0.0 tag since it will move forward with your new changes, so do be careful.

## Git Aliases

Before we move on to the next chapter, we want to introduce a feature that can make your Git 60 experience simpler, easier, and more familiar: aliases. For clarity's sake, we won't be using them anywhere else in this book, but if you go on to use Git with any regularity, aliases are something you should know about. Git doesn't automatically infer your command if you type it in partially. If you don't want to type the entire text of each of the Git commands, you can easily set up an alias for each command using git config. Here are a couple of examples you may want to set up:
$ git config --global alias.co checkout $ git config --global alias.br branch $ git config --global alias.ci commit $ git config --global alias.st status This means that, for example, instead of typing git commit, you just need to type git ci. As you go on using Git, you'll probably use other commands frequently as well; don't hesitate to create new aliases. This technique can also be very useful in creating commands that you think should exist. For example, to correct the usability problem you encountered with unstaging a file, you can add your own unstage alias to Git:
$ git config --global alias.unstage 'reset HEAD --'
This makes the following two commands equivalent:
$ git unstage fileA $ git reset HEAD -- fileA
This seems a bit clearer. It's also common to add a last command, like this:
$ git config --global alias.last 'log -1 HEAD'
This way, you can see the last commit easily:
$ git last commit 66938dae3329c7aebe598c2246a8e6af90d04646 Author: Josh Goebel <dreamer3@example.com> Date: Tue Aug 26 19:48:51 2008 +0800 Test for current head Signed-off-by: Scott Chacon <schacon@example.com>
As you can tell, Git simply replaces the new command with whatever you alias it for. However, maybe you want to run an external command, rather than a Git subcommand. In that case, you start the command with a ! character. This is useful if you write your own tools that work with a Git repository. We can demonstrate by aliasing git visual to run gitk:
$ git config --global alias.visual '!gitk'

## Summary

At this point, you can do all the basic local Git operations - creating or cloning a repository, making changes, staging and committing those changes, and viewing the history of all the changes the repository has been through. Next, we'll cover Git's killer feature: its branching model.

## Git Branching

Nearly every VCS has some form of branching support. Branching means you diverge from the main line of development and continue to do work without messing with that main line. In many VCS tools, this is a somewhat expensive process, often requiring you to create a new copy of your source code directory, which can take a long time for large projects. Some people refer to Git's branching model as its "killer feature," and it certainly sets Git apart in the VCS community. Why is it so special? The way Git branches is incredibly lightweight, making branching operations nearly instantaneous, and switching back and forth between branches generally just as fast. Unlike many other VCSs, Git encourages workflows that branch and merge often, even multiple times in a day. Understanding and mastering this feature gives you a powerful and unique tool and can entirely change the way that you develop.

## Branches In A Nutshell

To really understand the way Git does branching, we need to take a step back and examine how Git stores its data. As you may remember from What is Git?, Git doesn't store data as a series of changesets or differences, but instead as a series of *snapshots*. When you make a commit, Git stores a commit object that contains a pointer to the snapshot of the content you staged. This object also contains the author's name and email address, the message that you typed, and pointers to the commit or commits that directly came before this commit (its parent or parents): zero parents for the initial commit, one parent for a normal commit, and multiple parents for a commit that results from a merge of two or more branches. To visualize this, let's assume that you have a directory containing three files, and you stage them all and commit. Staging the files computes a checksum for each one (the SHA-1 hash we mentioned in What is Git?), stores that version of the file in the Git repository (Git refers to them as *blobs*), and adds that checksum to the staging area:
$ git add README test.rb LICENSE $ git commit -m 'Initial commit' When you create the commit by running git commit, Git checksums each subdirectory (in this case, just the root project directory) and stores them as a tree object in the Git repository. Git then creates a commit object that has the metadata and a pointer to the root project tree so it can re-create that snapshot when needed. Your Git repository now contains five objects: three *blobs* (each representing the contents of one of the three files), one *tree* that lists the contents of the directory and specifies which file names are stored as which blobs, and one *commit* with the pointer to that root tree and all the commit metadata.

![38_image_0.png](38_image_0.png)

## Figure 9. A Commit And Its Tree

If you make some changes and commit again, the next commit stores a pointer to the commit that came immediately before it.

![38_image_1.png](38_image_1.png)

A branch in Git is simply a lightweight movable pointer to one of these commits. The default branch nme in Git is master . As you start making commits, you're given a master branch that points to the last commit you made. Every time you commit, the master branch pointer moves forward automatically.

![38_image_2.png](38_image_2.png)

The "master" branch in Git is not a special branch. It is exactly like any other branch. The only reason nearly every repository has one is that the git init command creates it by default and most people don't bother to change it.

Figure 11. A branch and its commit history

![70_image_0.png](70_image_0.png)

## Creating A New Branch

What happens when you create a new branch? Well, doing so creates a new pointer for you to move around. Let's say you want to create a new branch called testing. You do this with the git branch command:
$ git branch testing This creates a new pointer to the same commit you're currently on.

![70_image_1.png](70_image_1.png)

Figure 12. Two branches pointing into the same series of commits
How does Git know what branch you're currently on? It keeps a special pointer called HEAD. Note that this is a lot different than the concept of HEAD in other VCSs you may be used to, such as Subversion or CVS. In Git, this is a pointer to the local branch you're currently on. In this case, you're still on master. The git branch command only *created* a new branch - it didn't switch to that

![71_image_0.png](71_image_0.png)

Figure 13. HEAD pointing to a branch You can easily see this by running a simple git log command that shows you where the branch pointers are pointing. This option is called --decorate.

$ git log --oneline --decorate f30ab (HEAD -> master, testing) Add feature \#32 - ability to add new formats to the central interface 34ac2 Fix bug \#1328 - stack overflow under certain conditions 98ca9 Initial commit You can see the master and testing branches that are right there next to the f30ab commit.

## Switching Branches

To switch to an existing branch, you run the git checkout command. Let's switch to the new testing branch:
$ git checkout testing This moves HEAD to point to the testing branch.

Figure 14. HEAD points to the current branch

![72_image_0.png](72_image_0.png)

What is the significance of that? Well, let's do another commit:

$ vim test.rb ![72_image_1.png](72_image_1.png)

Figure 15. The HEAD branch moves forward when a commit is made This is interesting, because now your testing branch has moved forward, but your master branch still points to the commit you were on when you ran git checkout to switch branches. Let's switch back to the master branch:
$ git checkout master git log *doesn't show all the branches all the time* If you were to run git log right now, you might wonder where the "testing" branch you just created went, as it would not appear in the output.

The branch hasn't disappeared; Git just doesn't know that you're interested in that branch and it is trying to show you what it thinks you're interested in. In other words, by default, git log will only show commit history below the branch you've checked out.

![73_image_0.png](73_image_0.png)

That command did two things. It moved the HEAD pointer back to point to the master branch, and it reverted the files in your working directory back to the snapshot that master points to. This also means the changes you make from this point forward will diverge from an older version of the project. It essentially rewinds the work you've done in your testing branch so you can go in a different direction.

Switching branches changes files in your working directory

![73_image_1.png](73_image_1.png)

It's important to note that when you switch branches in Git, files in your working directory will change. If you switch to an older branch, your working directory will be reverted to look like it did the last time you committed on that branch. If Git cannot do it cleanly, it will not let you switch at all.

Let's make a few changes and commit again:
$ vim test.rb $ git commit -a -m 'Make other changes' Now your project history has diverged (see Divergent history). You created and switched to a branch, did some work on it, and then switched back to your main branch and did other work. Both of those changes are isolated in separate branches: you can switch back and forth between the branches and merge them together when you're ready. And you did all that with simple branch,

![74_image_0.png](74_image_0.png)

## Figure 17. Divergent History

You can also see this easily with the git log command. If you run git log --oneline --decorate
--graph --all it will print out the history of your commits, showing where your branch pointers are and how your history has diverged.

$ git log --oneline --decorate --graph --all * c2b9e (HEAD, master) Make other changes | * 87ab2 (testing) Make a change |/
* f30ab Add feature \#32 - ability to add new formats to the central interface * 34ac2 Fix bug \#1328 - stack overflow under certain conditions * 98ca9 Initial commit of my project Because a branch in Git is actually a simple file that contains the 40 character SHA-1 checksum of the commit it points to, branches are cheap to create and destroy. Creating a new branch is as quick and simple as writing 41 bytes to a file (40 characters and a newline). This is in sharp contrast to the way most older VCS tools branch, which involves copying all of the project's files into a second directory. This can take several seconds or even minutes, depending on the size of the project, whereas in Git the process is always instantaneous. Also, because we're recording the parents when we commit, finding a proper merge base for merging is automatically done for us and is generally very easy to do. These features help encourage developers to create and use branches often. Let's see why you should do so.

Creating a new branch and switching to it at the same time

![75_image_0.png](75_image_0.png)

It's typical to create a new branch and want to switch to that new branch at the same time - this can be done in one operation with git checkout -b
<newbranchname>.

From Git version 2.23 onwards you can use git switch instead of git checkout to:

![75_image_1.png](75_image_1.png)

- Switch to an existing branch: git switch testing-branch.

- Create a new branch and switch to it: git switch -c new-branch. The -c flag stands for create, you can also use the full flag: --create.

- Return to your previously checked out branch: git switch -.

## Basic Branching And Merging

Let's go through a simple example of branching and merging with a workflow that you might use in the real world. You'll follow these steps:
1. Do some work on a website. 2. Create a branch for a new user story you're working on. 3. Do some work in that branch.

At this stage, you'll receive a call that another issue is critical and you need a hotfix. You'll do the following:
1. Switch to your production branch. 2. Create a branch to add the hotfix. 3. After it's tested, merge the hotfix branch, and push to production. 4. Switch back to your original user story and continue working.

## Basic Branching

First, let's say you're working on your project and have a couple of commits already on the master branch.

![75_image_2.png](75_image_2.png)

Figure 18. A simple commit history
70 You've decided that you're going to work on issue \#53 in whatever issue-tracking system your company uses. To create a new branch and switch to it at the same time, you can run the git checkout command with the -b switch:
$ git checkout -b iss53 Switched to a new branch "iss53" This is shorthand for:
$ git branch iss53

![76_image_0.png](76_image_0.png) $ git checkout iss53 Figure 19. Creating a new branch pointer You work on your website and do some commits. Doing so moves the iss53 branch forward, because you have it checked out (that is, your HEAD is pointing to it):
$ vim index.html $ git commit -a -m 'Create new footer [issue 53]'
Figure 20. The iss53 *branch has moved forward with your work*

![77_image_0.png](77_image_0.png)

Now you get the call that there is an issue with the website, and you need to fix it immediately. With Git, you don't have to deploy your fix along with the iss53 changes you've made, and you don't have to put a lot of effort into reverting those changes before you can work on applying your fix to what is in production. All you have to do is switch back to your master branch.

However, before you do that, note that if your working directory or staging area has uncommitted changes that conflict with the branch you're checking out, Git won't let you switch branches. It's best to have a clean working state when you switch branches. There are ways to get around this (namely, stashing and commit amending) that we'll cover later on, in Stashing and Cleaning. For now, let's assume you've committed all your changes, so you can switch back to your master branch:
$ git checkout master Switched to branch 'master' At this point, your project working directory is exactly the way it was before you started working on issue \#53, and you can concentrate on your hotfix. This is an important point to remember:
when you switch branches, Git resets your working directory to look like it did the last time you committed on that branch. It adds, removes, and modifies files automatically to make sure your working copy is what the branch looked like on your last commit to it.

Next, you have a hotfix to make. Let's create a hotfix branch on which to work until it's completed:
$ git checkout -b hotfix Switched to a new branch 'hotfix' $ vim index.html $ git commit -a -m 'Fix broken email address' [hotfix 1fb7853] Fix broken email address 1 file changed, 2 insertions(+)
Figure 21. Hotfix branch based on master

![78_image_0.png](78_image_0.png)

You can run your tests, make sure the hotfix is what you want, and finally merge the hotfix branch back into your master branch to deploy to production. You do this with the git merge command:
$ git checkout master $ git merge hotfix Updating f42c576..3a0874c Fast-forward index.html | 2 ++ 1 file changed, 2 insertions(+)
You'll notice the phrase "fast-forward" in that merge. Because the commit C4 pointed to by the branch hotfix you merged in was directly ahead of the commit C2 you're on, Git simply moves the pointer forward. To phrase that another way, when you try to merge one commit with a commit that can be reached by following the first commit's history, Git simplifies things by moving the pointer forward because there is no divergent work to merge together - this is called a "fastforward."
Your change is now in the snapshot of the commit pointed to by the master branch, and you can deploy the fix.

Figure 22. master *is fast-forwarded to* hotfix

![79_image_0.png](79_image_0.png)

![79_image_1.png](79_image_1.png)

After your super-important fix is deployed, you're ready to switch back to the work you were doing before you were interrupted. However, first you'll delete the hotfix branch, because you no longer need it - the master branch points at the same place. You can delete it with the -d option to git branch:
$ git branch -d hotfix Deleted branch hotfix (3a0874c).

Now you can switch back to your work-in-progress branch on issue \#53 and continue working on it.

$ git checkout iss53 Switched to branch "iss53" $ vim index.html $ git commit -a -m 'Finish the new footer [issue 53]' [iss53 ad82d7a] Finish the new footer [issue 53] 1 file changed, 1 insertion(+)

## Figure 23. Work Continues On Iss53

![80_Image_0.Png](80_Image_0.Png)

It's worth noting here that the work you did in your hotfix branch is not contained in the files in your iss53 branch. If you need to pull it in, you can merge your master branch into your iss53 branch by running git merge master, or you can wait to integrate those changes until you decide to pull the iss53 branch back into master later.

## Basic Merging

Suppose you've decided that your issue \#53 work is complete and ready to be merged into your master branch. In order to do that, you'll merge your iss53 branch into master, much like you merged your hotfix branch earlier. All you have to do is check out the branch you wish to merge into and then run the git merge command:
$ git checkout master Switched to branch 'master' $ git merge iss53 Merge made by the 'recursive' strategy. index.html | 1 + 1 file changed, 1 insertion(+)
This looks a bit different than the hotfix merge you did earlier. In this case, your development history has diverged from some older point. Because the commit on the branch you're on isn't a direct ancestor of the branch you're merging in, Git has to do some work. In this case, Git does a simple three-way merge, using the two snapshots pointed to by the branch tips and the common ancestor of the two.

![50_image_0.png](50_image_0.png)

Instead of just moving the branch pointer forward, Git creates a new snapshot that results from this three-way merge and automatically creates a new commit that points to it. This is referred to as a merge commit, and is special in that it has more than one parent.

![50_image_1.png](50_image_1.png)

Now that your work is merged in, you have no further need for the iss53 branch. You can close the issue in your issue-tracking system, and delete the branch:
$ git branch -d iss53

## Basic Merge Conflicts

Occasionally, this process doesn't go smoothly. If you changed the same part of the same file differently in the two branches you're merging, Git won't be able to merge them cleanly. If your fix for issue \#53 modified the same part of a file as the hotfix branch, you'll get a merge conflict that looks something like this:
$ git merge iss53 Auto-merging index.html CONFLICT (content): Merge conflict in index.html Automatic merge failed; fix conflicts and then commit the result.

Git hasn't automatically created a new merge commit. It has paused the process while you resolve the conflict. If you want to see which files are unmerged at any point after a merge conflict, you can run git status:
$ git status On branch master You have unmerged paths. (fix conflicts and run "git commit") Unmerged paths: (use "git add <file>..." to mark resolution) both modified: index.html no changes added to commit (use "git add" and/or "git commit -a")
Anything that has merge conflicts and hasn't been resolved is listed as unmerged. Git adds standard conflict-resolution markers to the files that have conflicts, so you can open them manually and resolve those conflicts. Your file contains a section that looks something like this:
<<<<<<< HEAD:index.html <div id="footer">contact : email.support@github.com</div> ======= <div id="footer"> please contact us at support@github.com </div> >>>>>>> iss53:index.html This means the version in HEAD (your master branch, because that was what you had checked out when you ran your merge command) is the top part of that block (everything above the =======),
while the version in your iss53 branch looks like everything in the bottom part. In order to resolve the conflict, you have to either choose one side or the other or merge the contents yourself. For instance, you might resolve this conflict by replacing the entire block with this:
<div id="footer"> please contact us at email.support@github.com </div>
This resolution has a little of each section, and the <<<<<<<, =======, and >>>>>>> lines have been completely removed. After you've resolved each of these sections in each conflicted file, run git add on each file to mark it as resolved. Staging the file marks it as resolved in Git.

If you want to use a graphical tool to resolve these issues, you can run git mergetool, which fires up an appropriate visual merge tool and walks you through the conflicts:
$ git mergetool This message is displayed because 'merge.tool' is not configured. See 'git mergetool --tool-help' or 'git help config' for more details. 'git mergetool' will now attempt to use one of the following tools: opendiff kdiff3 tkdiff xxdiff meld tortoisemerge gvimdiff diffuse diffmerge ecmerge p4merge araxis bc3 codecompare vimdiff emerge Merging: index.html Normal merge conflict for 'index.html': {local}: modified file {remote}: modified file Hit return to start merge resolution tool (opendiff):
If you want to use a merge tool other than the default (Git chose opendiff in this case because the command was run on macOS), you can see all the supported tools listed at the top after "one of the following tools." Just type the name of the tool you'd rather use.

If you need more advanced tools for resolving tricky merge conflicts, we cover

![83_image_0.png](83_image_0.png)

more on merging in Advanced Merging.

After you exit the merge tool, Git asks you if the merge was successful. If you tell the script that it was, it stages the file to mark it as resolved for you. You can run git status again to verify that all conflicts have been resolved:
$ git status On branch master All conflicts fixed but you are still merging. (use "git commit" to conclude merge) Changes to be committed: modified: index.html If you're happy with that, and you verify that everything that had conflicts has been staged, you can type git commit to finalize the merge commit. The commit message by default looks something like this:
Merge branch 'iss53' Conflicts: index.html \# \# It looks like you may be committing a merge. \# If this is not correct, please remove the file \# .git/MERGE_HEAD \# and try again. \# Please enter the commit message for your changes. Lines starting \# with '\#' will be ignored, and an empty message aborts the commit. \# On branch master \# All conflicts fixed but you are still merging. \# \# Changes to be committed: \# modified: index.html \#
If you think it would be helpful to others looking at this merge in the future, you can modify this commit message with details about how you resolved the merge and explain why you did the changes you made if these are not obvious.

## Branch Management

Now that you've created, merged, and deleted some branches, let's look at some branchmanagement tools that will come in handy when you begin using branches all the time.

The git branch command does more than just create and delete branches. If you run it with no arguments, you get a simple listing of your current branches:
$ git branch iss53 * master testing Notice the * character that prefixes the master branch: it indicates the branch that you currently have checked out (i.e., the branch that HEAD points to). This means that if you commit at this point, the master branch will be moved forward with your new work. To see the last commit on each branch, you can run git branch -v:
$ git branch -v iss53 93b412c Fix javascript issue * master 7a98805 Merge branch 'iss53' testing 782fd34 Add scott to the author list in the readme The useful --merged and --no-merged options can filter this list to branches that you have or have not yet merged into the branch you're currently on. To see which branches are already merged into the branch you're on, you can run git branch --merged:
$ git branch --merged iss53 * master Because you already merged in iss53 earlier, you see it in your list. Branches on this list without the
* in front of them are generally fine to delete with git branch -d; you've already incorporated their work into another branch, so you're not going to lose anything.

To see all the branches that contain work you haven't yet merged in, you can run git branch --no
-merged:
$ git branch --no-merged testing This shows your other branch. Because it contains work that isn't merged in yet, trying to delete it with git branch -d will fail:
$ git branch -d testing error: The branch 'testing' is not fully merged. If you are sure you want to delete it, run 'git branch -D testing'.

If you really do want to delete the branch and lose that work, you can force it with -D, as the helpful message points out.

The options described above, --merged and --no-merged will, if not given a commit or branch name as an argument, show you what is, respectively, merged or not merged into your *current* branch. You can always provide an additional argument to ask about the merge state with respect to some other branch without checking that other branch out first, as in, what is not merged into the master branch?

$ git checkout testing $ git branch --no-merged master topicA featureB

## Changing A Branch Name

![85_image_0.png](85_image_0.png)

Do not rename branches that are still in use by other collaborators. Do not rename a branch like master/main/mainline without having read the section Changing the master branch name.

Suppose you have a branch that is called bad-branch-name and you want to change it to correctedbranch-name, while keeping all history. You also want to change the branch name on the remote
(GitHub, GitLab, other server). How do you do this?

Rename the branch locally with the git branch --move command:
$ git branch --move bad-branch-name corrected-branch-name This replaces your bad-branch-name with corrected-branch-name, but this change is only local for now. To let others see the corrected branch on the remote, push it:
$ git push --set-upstream origin corrected-branch-name Now we'll take a brief look at where we are now:
$ git branch --all * corrected-branch-name main remotes/origin/bad-branch-name remotes/origin/corrected-branch-name remotes/origin/main Notice that you're on the branch corrected-branch-name and it's available on the remote. However, the branch with the bad name is also still present there but you can delete it by executing the following command:
$ git push origin --delete bad-branch-name Now the bad branch name is fully replaced with the corrected branch name. Changing the master branch name

![86_image_0.png](86_image_0.png)

Changing the name of a branch like master/main/mainline/default will break the integrations, services, helper utilities and build/release scripts that your repository uses. Before you do this, make sure you consult with your collaborators. Also, make sure you do a thorough search through your repo and update any references to the old branch name in your code and scripts.

Rename your local master branch into main with the following command:
$ git branch --move master main There's no local master branch anymore, because it's renamed to the main branch.

To let others see the new main branch, you need to push it to the remote. This makes the renamed
$ git push --set-upstream origin main Now we end up with the following state:
$ git branch --all * main remotes/origin/HEAD -> origin/master remotes/origin/main remotes/origin/master Your local master branch is gone, as it's replaced with the main branch. The main branch is present on the remote. However, the old master branch is still present on the remote. Other collaborators will continue to use the master branch as the base of their work, until you make some further changes.

Now you have a few more tasks in front of you to complete the transition:
- Any projects that depend on this one will need to update their code and/or configuration. - Update any test-runner configuration files. - Adjust build and release scripts. - Redirect settings on your repo host for things like the repo's default branch, merge rules, and other things that match branch names.

- Update references to the old branch in documentation. - Close or merge any pull requests that target the old branch.

After you've done all these tasks, and are certain the main branch performs just as the master branch, you can delete the master branch:
$ git push origin --delete master

## Branching Workflows

Now that you have the basics of branching and merging down, what can or should you do with them? In this section, we'll cover some common workflows that this lightweight branching makes possible, so you can decide if you would like to incorporate them into your own development cycle.

## Long-Running Branches

Because Git uses a simple three-way merge, merging from one branch into another multiple times over a long period is generally easy to do. This means you can have several branches that are always open and that you use for different stages of your development cycle; you can merge regularly from some of them into others.

82 Many Git developers have a workflow that embraces this approach, such as having only code that is entirely stable in their master branch - possibly only code that has been or will be released. They have another parallel branch named develop or next that they work from or use to test stability - it isn't necessarily always stable, but whenever it gets to a stable state, it can be merged into master.

It's used to pull in topic branches (short-lived branches, like your earlier iss53 branch) when they're ready, to make sure they pass all the tests and don't introduce bugs.

In reality, we're talking about pointers moving up the line of commits you're making. The stable branches are farther down the line in your commit history, and the bleeding-edge branches are farther up the history.

![88_image_0.png](88_image_0.png)

Figure 26. A linear view of progressive-stability branching
It's generally easier to think about them as work silos, where sets of commits graduate to a more stable silo when they're fully tested.

![88_image_1.png](88_image_1.png)

You can keep doing this for several levels of stability. Some larger projects also have a proposed or pu
(proposed updates) branch that has integrated branches that may not be ready to go into the next or master branch. The idea is that your branches are at various levels of stability; when they reach a more stable level, they're merged into the branch above them. Again, having multiple long-running branches isn't necessary, but it's often helpful, especially when you're dealing with very large or complex projects.

## Topic Branches

Topic branches, however, are useful in projects of any size. A topic branch is a short-lived branch that you create and use for a single particular feature or related work. This is something you've likely never done with a VCS before because it's generally too expensive to create and merge branches. But in Git it's common to create, work on, merge, and delete branches several times a day.

You saw this in the last section with the iss53 and hotfix branches you created. You did a few commits on them and deleted them directly after merging them into your main branch. This technique allows you to context-switch quickly and completely - because your work is separated into silos where all the changes in that branch have to do with that topic, it's easier to see what has happened during code review and such. You can keep the changes there for minutes, days, or months, and merge them in when they're ready, regardless of the order in which they were created or worked on.

Consider an example of doing some work (on master), branching off for an issue (iss91), working on it for a bit, branching off the second branch to try another way of handling the same thing (

![89_image_0.png](89_image_0.png)

iss91v2), going back to your master branch and working there for a while, and then branching off there to do some work that you're not sure is a good idea (dumbidea branch). Your commit history will look something like this:
Now, let's say you decide you like the second solution to your issue best (iss91v2); and you showed the dumbidea branch to your coworkers, and it turns out to be genius. You can throw away the original iss91 branch (losing commits C5 and C6) and merge in the other two. Your history then looks like this:
Figure 29. History after merging dumbidea *and* iss91v2

![90_image_0.png](90_image_0.png)

We will go into more detail about the various possible workflows for your Git project in Distributed Git, so before you decide which branching scheme your next project will use, be sure to read that chapter. It's important to remember when you're doing all this that these branches are completely local. When you're branching and merging, everything is being done only in your Git repository - there is no communication with the server.

## Remote Branches

Remote references are references (pointers) in your remote repositories, including branches, tags, and so on. You can get a full list of remote references explicitly with git ls-remote <remote>, or git remote show <remote> for remote branches as well as more information. Nevertheless, a more common way is to take advantage of remote-tracking branches. Remote-tracking branches are references to the state of remote branches. They're local references that you can't move; Git moves them for you whenever you do any network communication, to make sure they accurately represent the state of the remote repository. Think of them as bookmarks, to remind you where the branches in your remote repositories were the last time you connected to them.

Remote-tracking branch names take the form <remote>/<branch>. For instance, if you wanted to see what the master branch on your origin remote looked like as of the last time you communicated with it, you would check the origin/master branch. If you were working on an issue with a partner and they pushed up an iss53 branch, you might have your own local iss53 branch, but the branch on the server would be represented by the remote-tracking branch origin/iss53.

This may be a bit confusing, so let's look at an example. Let's say you have a Git server on your network at git.ourcompany.com. If you clone from this, Git's clone command automatically names it origin for you, pulls down all its data, creates a pointer to where its master branch is, and names it origin/master locally. Git also gives you your own local master branch starting at the same place as origin's master branch, so you have something to work from.

## "Origin" Is Not Special

![91_image_0.png](91_image_0.png)

Just like the branch name "master" does not have any special meaning in Git, neither does "origin". While "master" is the default name for a starting branch when you run git init which is the only reason it's widely used, "origin" is the default name for a remote when you run git clone. If you run git clone -o booyah instead, then you will have booyah/master as your default remote branch.

![61_image_0.png](61_image_0.png)

If you do some work on your local master branch, and, in the meantime, someone else pushes to git.ourcompany.com and updates its master branch, then your histories move forward differently.

Also, as long as you stay out of contact with your origin server, your origin/master pointer doesn't move.

![62_image_0.png](62_image_0.png)

![62_image_1.png](62_image_1.png)

To synchronize your work with a given remote, you run a git fetch <remote> command (in our case, git fetch origin ). This command looks up which server "origin" is (in this case, it's git.ourcompany.com ), fetches any data from it that you don't yet have, and updates your local database, moving your origin/master pointer to its new, more up-to-date position.

![63_image_0.png](63_image_0.png)

To demonstrate having multiple remote servers and what remote branches for those remote projects look like, let's assume you have another internal Git server that is used only for development by one of your sprint teams. This server is at git.team1.ourcompany.com . You can add it as a new remote reference to the project you're currently working on by running the git remote add command as we covered in Git Basics. Name this remote teamone , which will be your shortname for that whole URL.

![64_image_0.png](64_image_0.png)

Now, you can run git fetch teamone to fetch everything the remote teamone server has that you don't have yet. Because that server has a subset of the data your origin server has right now, Git fetches no data but sets a remote-tracking branch called teamone/master to point to the commit that teamone has as its master branch.

![96_image_0.png](96_image_0.png)

## Pushing

When you want to share a branch with the world, you need to push it up to a remote to which you have write access. Your local branches aren't automatically synchronized to the remotes you write to - you have to explicitly push the branches you want to share. That way, you can use private branches for work you don't want to share, and push up only the topic branches you want to collaborate on.

If you have a branch named serverfix that you want to work on with others, you can push it up the same way you pushed your first branch. Run git push <remote> <branch>:
$ git push origin serverfix Counting objects: 24, done. Delta compression using up to 8 threads. Compressing objects: 100% (15/15), done. Writing objects: 100% (24/24), 1.91 KiB | 0 bytes/s, done. Total 24 (delta 2), reused 0 (delta 0) To https://github.com/schacon/simplegit * [new branch] serverfix -> serverfix This is a bit of a shortcut. Git automatically expands the serverfix branchname out to refs/heads/serverfix:refs/heads/serverfix, which means, "Take my serverfix local branch and push it to update the remote's serverfix branch." We'll go over the refs/heads/ part in detail in Git Internals, but you can generally leave it off. You can also do git push origin serverfix:serverfix, which does the same thing - it says, "Take my serverfix and make it the remote's serverfix." You can use this format to push a local branch into a remote branch that is named differently. If you didn't want it to be called serverfix on the remote, you could instead run git push origin serverfix:awesomebranch to push your local serverfix branch to the awesomebranch branch on the remote project.

Don't type your password every time If you're using an HTTPS URL to push over, the Git server will ask you for your username and password for authentication. By default it will prompt you on the terminal for this information so the server can tell if you're allowed to push.

![97_image_0.png](97_image_0.png)

If you don't want to type it every single time you push, you can set up a "credential cache". The simplest is just to keep it in memory for a few minutes, which you can easily set up by running git config --global credential.helper cache.

For more information on the various credential caching options available, see Credential Storage.

![97_image_3.png](97_image_3.png)

![97_image_4.png](97_image_4.png)

The next time one of your collaborators fetches from the server, they will get a reference to where the server's version of serverfix is under the remote branch origin/serverfix:
$ git fetch origin

![97_image_1.png](97_image_1.png)

![97_image_2.png](97_image_2.png) remote: Counting objects: 7, done. remote: Compressing objects: 100% (2/2), done. remote: Total 3 (delta 0), reused 3 (delta 0) Unpacking objects: 100% (3/3), done. From https://github.com/schacon/simplegit * [new branch] serverfix -> origin/serverfix It's important to note that when you do a fetch that brings down new remote-tracking branches, you don't automatically have local, editable copies of them. In other words, in this case, you don't have a new serverfix branch - you have only an origin/serverfix pointer that you can't modify.

To merge this work into your current working branch, you can run git merge origin/serverfix. If you want your own serverfix branch that you can work on, you can base it off your remotetracking branch:
$ git checkout -b serverfix origin/serverfix Branch serverfix set up to track remote branch serverfix from origin. Switched to a new branch 'serverfix' This gives you a local branch that you can work on that starts where origin/serverfix is.

## Tracking Branches

Checking out a local branch from a remote-tracking branch automatically creates what is called a "tracking branch" (and the branch it tracks is called an "upstream branch"). Tracking branches are local branches that have a direct relationship to a remote branch. If you're on a tracking branch and type git pull, Git automatically knows which server to fetch from and which branch to merge in.

When you clone a repository, it generally automatically creates a master branch that tracks origin/master. However, you can set up other tracking branches if you wish - ones that track branches on other remotes, or don't track the master branch. The simple case is the example you just saw, running git checkout -b <branch> <remote>/<branch>. This is a common enough operation that Git provides the --track shorthand:
$ git checkout --track origin/serverfix Branch serverfix set up to track remote branch serverfix from origin. Switched to a new branch 'serverfix' In fact, this is so common that there's even a shortcut for that shortcut. If the branch name you're trying to checkout (a) doesn't exist and (b) exactly matches a name on only one remote, Git will create a tracking branch for you:
$ git checkout serverfix Branch serverfix set up to track remote branch serverfix from origin. Switched to a new branch 'serverfix' To set up a local branch with a different name than the remote branch, you can easily use the first version with a different local branch name:
$ git checkout -b sf origin/serverfix Branch sf set up to track remote branch serverfix from origin.

Switched to a new branch 'sf' Now, your local branch sf will automatically pull from origin/serverfix.

If you already have a local branch and want to set it to a remote branch you just pulled down, or want to change the upstream branch you're tracking, you can use the -u or --set-upstream-to option to git branch to explicitly set it at any time.

$ git branch -u origin/serverfix Branch serverfix set up to track remote branch serverfix from origin.

Upstream shorthand

![98_image_0.png](98_image_0.png)

When you have a tracking branch set up, you can reference its upstream branch with the @{upstream} or @{u} shorthand. So if you're on the master branch and it's tracking origin/master, you can say something like git merge @{u} instead of git merge origin/master if you wish.

If you want to see what tracking branches you have set up, you can use the -vv option to git branch.

This will list out your local branches with more information including what each branch is tracking and if your local branch is ahead, behind or both.

$ git branch -vv iss53 7e424c3 [origin/iss53: ahead 2] Add forgotten brackets master 1ae2a45 [origin/master] Deploy index fix * serverfix f8674d9 [teamone/server-fix-good: ahead 3, behind 1] This should do it testing 5ea463a Try something new So here we can see that our iss53 branch is tracking origin/iss53 and is "ahead" by two, meaning that we have two commits locally that are not pushed to the server. We can also see that our master branch is tracking origin/master and is up to date. Next we can see that our serverfix branch is tracking the server-fix-good branch on our teamone server and is ahead by three and behind by one, meaning that there is one commit on the server we haven't merged in yet and three commits locally that we haven't pushed. Finally we can see that our testing branch is not tracking any remote branch. It's important to note that these numbers are only since the last time you fetched from each server. This command does not reach out to the servers, it's telling you about what it has cached from these servers locally. If you want totally up to date ahead and behind numbers, you'll need to fetch from all your remotes right before running this. You could do that like this:
$ git fetch --all; git branch -vv

## Pulling

While the git fetch command will fetch all the changes on the server that you don't have yet, it will not modify your working directory at all. It will simply get the data for you and let you merge it yourself. However, there is a command called git pull which is essentially a git fetch immediately followed by a git merge in most cases. If you have a tracking branch set up as demonstrated in the last section, either by explicitly setting it or by having it created for you by the clone or checkout commands, git pull will look up what server and branch your current branch is tracking, fetch from that server and then try to merge in that remote branch.

Generally it's better to simply use the fetch and merge commands explicitly as the magic of git pull can often be confusing.

## Deleting Remote Branches

Suppose you're done with a remote branch - say you and your collaborators are finished with a feature and have merged it into your remote's master branch (or whatever branch your stable codeline is in). You can delete a remote branch using the --delete option to git push. If you want to delete your serverfix branch from the server, you run the following:
$ git push origin --delete serverfix To https://github.com/schacon/simplegit - [deleted] serverfix Basically all this does is to remove the pointer from the server. The Git server will generally keep the data there for a while until a garbage collection runs, so if it was accidentally deleted, it's often easy to recover.

## Rebasing

In Git, there are two main ways to integrate changes from one branch into another: the merge and the rebase. In this section you'll learn what rebasing is, how to do it, why it's a pretty amazing tool, and in what cases you won't want to use it.

## The Basic Rebase

If you go back to an earlier example from Basic Merging, you can see that you diverged your work and made commits on two different branches.

![100_image_0.png](100_image_0.png)

Figure 35. Simple divergent history The easiest way to integrate the branches, as we've already covered, is the merge command. It performs a three-way merge between the two latest branch snapshots (C3 and C4) and the most recent common ancestor of the two (C2), creating a new snapshot (and commit).

Figure 36. Merging to integrate diverged work history

![101_image_0.png](101_image_0.png)

However, there is another way: you can take the patch of the change that was introduced in C4 and reapply it on top of C3. In Git, this is called *rebasing*. With the rebase command, you can take all the changes that were committed on one branch and replay them on a different branch.

For this example, you would check out the experiment branch, and then rebase it onto the master branch as follows:
$ git checkout experiment $ git rebase master First, rewinding head to replay your work on top of it... Applying: added staged command This operation works by going to the common ancestor of the two branches (the one you're on and the one you're rebasing onto), getting the diff introduced by each commit of the branch you're on, saving those diffs to temporary files, resetting the current branch to the same commit as the branch you are rebasing onto, and finally applying each change in turn.

![101_image_1.png](101_image_1.png)

![101_image_2.png](101_image_2.png)

Figure 37. Rebasing the change introduced in C4 *onto* C3 At this point, you can go back to the master branch and do a fast-forward merge.

$ git checkout master $ git merge experiment Figure 38. Fast-forwarding the master *branch*

![102_image_0.png](102_image_0.png)

Now, the snapshot pointed to by C4' is exactly the same as the one that was pointed to by C5 in the merge example. There is no difference in the end product of the integration, but rebasing makes for a cleaner history. If you examine the log of a rebased branch, it looks like a linear history: it appears that all the work happened in series, even when it originally happened in parallel. Often, you'll do this to make sure your commits apply cleanly on a remote branch - perhaps in a project to which you're trying to contribute but that you don't maintain. In this case, you'd do your work in a branch and then rebase your work onto origin/master when you were ready to submit your patches to the main project. That way, the maintainer doesn't have to do any integration work - just a fast-forward or a clean apply. Note that the snapshot pointed to by the final commit you end up with, whether it's the last of the rebased commits for a rebase or the final merge commit after a merge, is the same snapshot - it's only the history that is different. Rebasing replays changes from one line of work onto another in the order they were introduced, whereas merging takes the endpoints and merges them together.

## More Interesting Rebases

You can also have your rebase replay on something other than the rebase target branch. Take a history like A history with a topic branch off another topic branch, for example. You branched a topic branch (server) to add some server-side functionality to your project, and made a commit.

Then, you branched off that to make the client-side changes (client) and committed a few times. Finally, you went back to your server branch and did a few more commits.

![72_image_0.png](72_image_0.png)

Suppose you decide that you want to merge your client-side changes into your mainline for a release, but you want to hold off on the server-side changes until it's tested further. You can take the changes on client that aren't on server (C8 and C9) and replay them on your master branch by using the --onto option of git  rebase:
$ git rebase --onto master server client This basically says, "Take the client branch, figure out the patches since it diverged from the server branch, and replay these patches in the client branch as if it was based directly off the master branch instead." It's a bit complex, but the result is pretty cool.

![72_image_1.png](72_image_1.png)

Now you can fast-forward your master branch (see Fast-forwarding your master branch to include the client branch changes):
$ git checkout master

![104_image_0.png](104_image_0.png) $ git merge client Figure 41. Fast-forwarding your master *branch to include the* client *branch changes* Let's say you decide to pull in your server branch as well. You can rebase the server branch onto the master branch without having to check it out first by running git rebase <basebranch> <topicbranch> - which checks out the topic branch (in this case, server) for you and replays it onto the base branch (master):
$ git rebase master server This replays your server work on top of your master work, as shown in Rebasing your server branch on top of your master branch.

![104_image_1.png](104_image_1.png)

Figure 42. Rebasing your server *branch on top of your* master *branch* Then, you can fast-forward the base branch (master):
$ git checkout master $ git merge server You can remove the client and server branches because all the work is integrated and you don't need them anymore, leaving your history for this entire process looking like Final commit history:
$ git branch -d client $ git branch -d server Figure 43. Final commit history

![105_image_0.png](105_image_0.png)

![105_image_1.png](105_image_1.png)

## The Perils Of Rebasing

Ahh, but the bliss of rebasing isn't without its drawbacks, which can be summed up in a single line: Do not rebase commits that exist outside your repository and that people may have based work on. If you follow that guideline, you'll be fine. If you don't, people will hate you, and you'll be scorned by friends and family. When you rebase stuff, you're abandoning existing commits and creating new ones that are similar but different. If you push commits somewhere and others pull them down and base work on them, and then you rewrite those commits with git rebase and push them up again, your collaborators will have to re-merge their work and things will get messy when you try to pull their work back into yours. Let's look at an example of how rebasing work that you've made public can cause problems. Suppose you clone from a central server and then do some work off that. Your commit history looks like this:

![105_image_2.png](105_image_2.png)

![105_image_3.png](105_image_3.png)

Figure 44. Clone a repository, and base some work on it Now, someone else does more work that includes a merge, and pushes that work to the central server. You fetch it and merge the new remote branch into your work, making your history look something like this:

![75_image_0.png](75_image_0.png)

![75_image_1.png](75_image_1.png)

![75_image_2.png](75_image_2.png)

Next, the person who pushed the merged work decides to go back and rebase their work instead; they do a git push --force to overwrite the history on the server. You then fetch from that server, bringing down the new commits.

![75_image_4.png](75_image_4.png)

![75_image_3.png](75_image_3.png)

![75_image_5.png](75_image_5.png)

Now you're both in a pickle. If you do a git pull, you'll create a merge commit which includes both lines of history, and your repository will look like this:

![107_image_0.png](107_image_0.png)

![107_image_1.png](107_image_1.png)

![107_image_2.png](107_image_2.png)

![107_image_3.png](107_image_3.png)

![107_image_4.png](107_image_4.png)

If you run a git log when your history looks like this, you'll see two commits that have the same author, date, and message, which will be confusing. Furthermore, if you push this history back up to the server, you'll reintroduce all those rebased commits to the central server, which can further confuse people. It's pretty safe to assume that the other developer doesn't want C4 and C6 to be in the history; that's why they rebased in the first place.

## Rebase When You Rebase

If you do find yourself in a situation like this, Git has some further magic that might help you out. If someone on your team force pushes changes that overwrite work that you've based work on, your challenge is to figure out what is yours and what they've rewritten. It turns out that in addition to the commit SHA-1 checksum, Git also calculates a checksum that is based just on the patch introduced with the commit. This is called a "patch-id". If you pull down work that was rewritten and rebase it on top of the new commits from your partner, Git can often successfully figure out what is uniquely yours and apply them back on top of the new branch. For instance, in the previous scenario, if instead of doing a merge when we're at Someone pushes rebased commits, abandoning commits you've based your work on we run git rebase teamone/master, Git will:
- Determine what work is unique to our branch (C2, C3, C4, C6, C7)
- Determine which are not merge commits (C2, C3, C4)
- Determine which have not been rewritten into the target branch (just C2 and C3, since C4 is the same patch as C4')
So instead of the result we see in You merge in the same work again into a new merge commit, we would end up with something more like Rebase on top of force-pushed rebase work.

![108_image_0.png](108_image_0.png)

![108_image_1.png](108_image_1.png)

This only works if C4 and C4' that your partner made are almost exactly the same patch. Otherwise the rebase won't be able to tell that it's a duplicate and will add another C4-like patch (which will probably fail to apply cleanly, since the changes would already be at least somewhat there).

You can also simplify this by running a git pull --rebase instead of a normal git pull. Or you could do it manually with a git fetch followed by a git rebase teamone/master in this case.

If you are using git pull and want to make --rebase the default, you can set the pull.rebase config value with something like git config --global pull.rebase true.

If you only ever rebase commits that have never left your own computer, you'll be just fine. If you rebase commits that have been pushed, but that no one else has based commits from, you'll also be fine. If you rebase commits that have already been pushed publicly, and people may have based work on those commits, then you may be in for some frustrating trouble, and the scorn of your teammates.

If you or a partner does find it necessary at some point, make sure everyone knows to run git pull
--rebase to try to make the pain after it happens a little bit simpler.

## Rebase Vs. Merge

Now that you've seen rebasing and merging in action, you may be wondering which one is better. Before we can answer this, let's step back a bit and talk about what history means. One point of view on this is that your repository's commit history is a record of what actually happened. It's a historical document, valuable in its own right, and shouldn't be tampered with. From this angle, changing the commit history is almost blasphemous; you're *lying* about what actually transpired. So what if there was a messy series of merge commits? That's how it happened, and the repository should preserve that for posterity. The opposing point of view is that the commit history is the **story of how your project was made.** You wouldn't publish the first draft of a book, so why show your messy work? When you're working on a project, you may need a record of all your missteps and dead-end paths, but when it's time to show your work to the world, you may want to tell a more coherent story of how to get from A to B. People in this camp use tools like rebase and filter-branch to rewrite their commits before they're merged into the mainline branch. They use tools like rebase and filter-branch, to tell the story in the way that's best for future readers. Now, to the question of whether merging or rebasing is better: hopefully you'll see that it's not that simple. Git is a powerful tool, and allows you to do many things to and with your history, but every team and every project is different. Now that you know how both of these things work, it's up to you to decide which one is best for your particular situation. You can get the best of both worlds: rebase local changes before pushing to clean up your work, but never rebase anything that you've pushed somewhere.

## Summary

We've covered basic branching and merging in Git. You should feel comfortable creating and switching to new branches, switching between branches and merging local branches together. You should also be able to share your branches by pushing them to a shared server, working with others on shared branches and rebasing your branches before they are shared. Next, we'll cover what you'll need to run your own Git repository-hosting server.