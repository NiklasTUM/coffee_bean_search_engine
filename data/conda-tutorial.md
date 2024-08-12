CHAPTER 1 User guide

## 1.1 **Overview**

This page provides an overview of how to use conda. For an overview of what conda is and what it does, please see the *front page*.

The quickest way to start using conda is to go through the 20-minute *Getting started with conda* guide.

The conda command is the primary interface for managing installations of various packages. It can:
- Query and search the Anaconda package index and current Anaconda installation.

- Create new conda environments.

- Install and update packages into existing conda environments.

TIP: You can abbreviate many frequently used command options that are preceded by 2 dashes (--) to just 1 dash and the first letter of the option. So --name and -n are the same, and --envs and -e are the same.

For full usage of each command, including abbreviations, see *Command reference*. You can see the same information at the command line by *viewing the command-line help*.

## 1.2 **Concepts**

- *Conda directory structure*
- *Conda environments* - *Conda packages*

## 1.2.1 **Conda Directory Structure**

This section describes the conda system directory structure.

ROOT_DIR The directory that Anaconda or Miniconda was installed into.

EXAMPLES:

| /opt/Anaconda   | #Linux   |
|-----------------|----------|
| C:\Anaconda     | #Windows |

## /Pkgs

Also referred to as PKGS_DIR. This directory contains decompressed packages, ready to be linked in conda environments. Each package resides in a subdirectory corresponding to its canonical name.

/envs The system location for additional conda environments to be created. The following subdirectories comprise the default Anaconda environment: /bin
/include
/lib /share Other conda environments usually contain the same subdirectories as the default environment.

## 1.2.2 **Conda Environments**

A conda environment is a directory that contains a specific collection of conda packages that you have installed.

For example, you may have one environment with NumPy 1.7 and its dependencies, and another environment with NumPy 1.6 for legacy testing. If you change one environment, your other environments are not affected. You can easily activate or deactivate environments, which is how you switch between them. You can also share your environment with someone by giving them a copy of your environment.yaml file. For more information, see Managing environments.

## 1.2.3 **Conda Packages**

A conda package is a compressed tarball file that contains system-level libraries, Python or other modules, executable programs and other components. Conda keeps track of the dependencies between packages and platforms.

Conda packages are downloaded from remote channels, which are URLs to directories containing conda packages.

The conda command searches a default set of channels, and packages are automatically downloaded and updated from http://repo.continuum.io/pkgs/. You can modify what remote channels are automatically searched. You might want to do this to maintain a private or internal channel. For details, see *Channel locations (channels)*. See also Managing packages.

The conda package format is identical across platforms and operating systems.

To install conda packages, in the Terminal or an Anaconda Prompt, run:
conda install [packagename]
NOTE: Replace [packagename] with the desired package name.

A conda package includes a link to a tarball or bzipped tar archive, with the extension ".tar.bz2", which contains metadata under the info/ directory and a collection of files that are installed directly into an install prefix.

During the install process, files are extracted into the install prefix, except for files in the info/ directory. Installing the files of a conda package into an environment can be thought of as changing the directory to an environment, and then downloading and extracting the .zip file and its dependencies—all with the single conda install
[packagename] command.

## 1.3 **Getting Started With Conda**

Conda is a powerful package manager and environment manager that you use with command line commands at the Anaconda Prompt for Windows, or in a Terminal window for macOS or Linux.

This 20-minute guide to getting started with conda lets you try out the major features of conda. You should understand how conda works when you finish this guide.

SEE ALSO: Getting started with Anaconda Navigator, a graphical user interface that lets you use conda in a weblike interface without having to enter manual commands. Compare the Getting started guides for each to see which program you prefer.

## 1.3.1 **Before You Start**

You should have already installed Anaconda.

## 1.3.2 **Contents**

- *Starting conda* on Windows, macOS or Linux. 2 MINUTES
- *Managing conda*. Verify that Anaconda is installed and check that conda is updated to the current version. 3 MINUTES
- *Managing environments*. Create *environments* and move easily between them. 5 MINUTES - *Managing Python*. Create an environment that has a different version of Python. 5 MINUTES
- *Managing packages*. Find packages available for you to install. Install packages. 5 MINUTES
TOTAL TIME: 20 MINUTES

## 1.3.3 **Starting Conda** Windows

- From the Start menu, search for and open "Anaconda Prompt".

![9_image_0.png](9_image_0.png)

On Windows, all commands below are typed into the Anaconda Prompt window.

MacOS
- Open Launchpad, then click the Terminal icon.

On macOS, all commands below are typed into the Terminal window.

Linux
- Open a Terminal window.

On Linux, all commands below are typed into the Terminal window.

## 1.3.4 **Managing Conda**

Verify that conda is installed and running on your system by typing:
conda --version Conda displays the number of the version that you have installed. You do not need to navigate to the Anaconda directory.

EXAMPLE: conda 4.4.9 NOTE: If you get an error message, make sure you closed and re-opened the Terminal window after installing, or do it now. Then verify that you are logged into the same user account that you used to install Anaconda or Miniconda.

Update conda to the current version. Type the following:
conda update conda Conda compares versions and then displays what is available to install.

If a newer version of conda is available, type y to update:
Proceed ([y]/n)? y TIP: We recommend that you always keep conda updated to the latest version.

## 1.3.5 **Managing Environments**

Conda allows you to to create separate environments containing files, packages and their dependencies that will not interact with other environments.

When you begin using conda, you already have a default environment named base. You don't want to put programs into your base environment, though. Create separate environments to keep your programs isolated from each other.

1. Create a new environment and install a package in it.

We will name the environment snowflakes and install the package BioPython. At the Anaconda Prompt or in your Terminal window, type the following:
conda create --name snowflakes biopython Conda checks to see what additional packages ("dependencies") Biopython will need, and asks if you want to proceed:
Proceed ([y]/n)? y Type "y" and press Enter to proceed.

2. To use, or "activate" the new environment, type the following:
- Windows: activate snowflakes
- Linux and macOS: source activate snowflakes Now that you are in your snowflakes environment, any conda commands you type will go to that environment until you deactivate it.

3. To see a list of all your environments, type:
conda info --envs A list of environments appears, similar to the following:
conda environments:
base /home/username/Anaconda3 snowflakes * /home/username/Anaconda3/envs/snowflakes TIP: The active environment is the one with an asterisk (*).

4. Change your current environment back to the default (base):
- Windows: deactivate - Linux, macOS: source deactivate TIP: When the environment is deactivated, its name is no longer shown in your prompt, and the asterisk (*)
returns to base. To verify, you can repeat the conda info --envs command.

## 1.3.6 **Managing Python**

When you create a new environment, conda installs the same Python version you used when you downloaded and installed Anaconda. If you want to use a different version of Python, for example Python 3.5, simply create a new environment and specify the version of Python that you want.

1. Create a new environment named "snakes" that contains Python 3.5:
conda create --name snakes python=3.5 When conda asks if you want to proceed, type "y" and press Enter.

2. Activate the new environment:
- Windows: activate snakes - Linux, macOS: source activate snakes 3. Verify that the snakes environment has been added and is active:
conda info --envs Conda displays the list of all environments with an asterisk (*) after the name of the active environment:
\# conda environments: \#
base /home/username/anaconda3 snakes * /home/username/anaconda3/envs/snakes snowflakes /home/username/anaconda3/envs/snowflakes The active environment is also displayed in front of your prompt in (parentheses) or [brackets] like this:
(snakes) $
4. Verify which version of Python is in your current environment:
python --version 5. Deactivate the snakes environment and return to base environment:
- Windows: deactivate
- Linux, macOS: source deactivate

## 1.3.7 **Managing Packages**

In this section, you check which packages you have installed, check which are available and look for a specific package and install it.

1. To find a package you have already installed, first activate the environment you want to search. Look above for the commands to *activate your snakes environment*.

2. Check to see if a package you have not installed named "beautifulsoup4" is available from the Anaconda repository (must be connected to the Internet):
conda search beautifulsoup4 Conda displays a list of all packages with that name on the Anaconda repository, so we know it is available.

3. Install this package into the current environment:
conda install beautifulsoup4 4. Check to see if the newly installed program is in this environment:
conda list

## 1.3.8 **More Information**

- *Conda cheat sheet*.

- Full documentation— https://conda.io/docs/ .

- Free community support— https://groups.google.com/a/anaconda.com/forum/\#!forum/anaconda . - Paid support options— https://www.anaconda.com/support/ .

## 1.4 **Installation**

- *System requirements* - *Regular installation*
- *Installing in silent mode*
- *Installing conda on a system that has other Python installations or packages* The fastest way to *obtain* conda is to install *Miniconda*, a mini version of *Anaconda* that includes only conda and its dependencies. If you prefer to have conda plus over 720 open source packages, install Anaconda.

We recommend you install Anaconda for the local user, which does not require administrator permissions and is the most robust type of installation. You can also install Anaconda system wide, which does require administrator permissions.

For information on using our graphical installers for Windows or macOS, see the instructions for installing Anaconda.

## 1.4.1 **System Requirements**

- 32- or 64-bit computer.

- For Miniconda—400 MB disk space.

- For Anaconda—Minimum 3 GB disk space to download and install. - Windows, macOS or Linux.

- Python 2.7, 3.4, 3.5 or 3.6.

- pycosat. - PyYaml.

- Requests.

NOTE: You do not need administrative or root permissions to install Anaconda if you select a user-writable install location.

## 1.4.2 **Regular Installation**

Follow the instructions for your operating system:
- *Windows*.

- *macOS*. - *Linux*.

## 1.4.3 **Installing In Silent Mode**

You can use *silent installation* of Miniconda or Anaconda for deployment or testing or building services such as Travis CI and AppVeyor.

Follow the silent-mode instructions for your operating system:
- *Windows*. - *macOS*.

- *Linux*.

## 1.4.4 **Installing Conda On A System That Has Other Python Installations Or Packages**

You do not need to uninstall other Python installations or packages in order to use conda. Even if you already have a system Python, another Python installation from a source such as the macOS Homebrew package manager and globally installed packages from pip such as pandas and NumPy, you do not need to uninstall, remove, or change any of them before using conda.

Install Anaconda or Miniconda normally, and let the installer add the conda installation of Python to your PATH
environment variable. There is no need to set the PYTHONPATH environment variable. To see if the conda installation of Python is in your PATH variable:
- On macOS and Linux, open the Terminal and run—echo $PATH.

- On Windows, open an Anaconda Prompt and run—echo %PATH%.

To see which Python installation is currently set as the default:
- On macOS and Linux, open the Terminal and run—which python.

- On Windows, open an Anaconda Prompt and run—where python.

To see which packages are installed in your current conda environment and their version numbers, in your Terminal window or an Anaconda Prompt, run conda list.

## Downloading Conda

- *Anaconda or Miniconda?*
- *Choosing a version of Anaconda or Miniconda* - *GUI versus command line installer*
- *Choosing a version of Python*
- *Cryptographic hash verification* You have 3 conda download options:
- Download Anaconda—free.

- Download Miniconda—free.

- Purchase Anaconda Enterprise.

You can download any of these 3 options with legacy Python 2.7 or current Python 3.

You can also choose a version with a GUI or a command line installer.

TIP: If you are unsure of which option to download, choose the most recent version of Anaconda3, which includes Python 3.6, the most recent version of Python. If you are on Windows or macOS, choose the version with the GUI
installer.

## Anaconda Or Miniconda?

Choose Anaconda if you:
- Are new to conda or Python. - Like the convenience of having Python and over 150 scientific packages automatically installed at once.

- Have the time and disk space—a few minutes and 300 MB.

- Do not want to individually install each of the packages you want to use.

Choose Miniconda if you:
- Do not mind installing each of the packages you want to use individually.

- Do not have time or disk space to install over 150 packages at once. - Want fast access to Python and the conda commands and you wish to sort out the other programs later.

## Choosing A Version Of Anaconda Or Miniconda

- Whether you use Anaconda or Miniconda, select the most recent version. - Select an older version from the archive only if you are testing or need an older version for a specific purpose.

- To use conda on Windows XP, select Anaconda 2.3.0 and see Using conda on Windows XP with or without a proxy.

## Gui Versus Command Line Installer

Both GUI and command line installers are available for Windows, macOS and Linux:
- If you do not wish to enter commands in a Terminal window, choose the GUI installer.

- If GUIs slow you down, choose the command line version.

## Choosing A Version Of Python

- The last version of Python 2 is 2.7, which is included with Anaconda and Miniconda. - The newest stable version of Python is 3.6, which is included with Anaconda3 and Miniconda3. - You can easily set up additional versions of Python such as 3.5 by downloading any version and creating a new environment with just a few clicks. See *Getting started with conda*.

## Cryptographic Hash Verification

MD5 checksums are available for Miniconda and both MD5 and SHA-256 checksums are available for Anaconda.

Download the installer file and before installing verify it as follows:
- macOS: In iTerm or a Terminal window enter md5 filename or shasum -a 256 filename.

NOTE: Replace filename with the actual path and name of the downloaded installer file.

- Linux: In a Terminal window enter md5sum filename or sha256sum filename.

NOTE: Replace filename with the actual path and name of the downloaded installer file.

- Windows:
- If you have PowerShell V4 or later:
Open a PowerShell console and verify the file as follows:
Get-FileHash filename -Algorithm MD5 or:
Get-FileHash filename -Algorithm SHA256 NOTE: Replace "filename" with the actual path and name of the downloaded file.

- If you don't have PowerShell V4 or later:
Use the free online verifier tool on the Microsoft website.

1. Download the file and extract it.

2. Open a Command Prompt window.

3. Navigate to the file. 4. Run one of the following commands:
* For MD5:
Start-PsFCIV -Path C:\path\to\file.ext -HashAlgorithm MD5 -Online
* For SHA256:
Start-PsFCIV -Path C:\path\to\file.ext -HashAlgorithm SHA256 -Online NOTE: In both commands, replace C:\path\to\file.ext with the actual path, filename and extension.

## Installing On Windows

1. Download the installer:
- Miniconda installer for Windows.

- Anaconda installer for Windows.

2. Double-click the .exe file.

3. Follow the instructions on the screen.

If you are unsure about any setting, accept the defaults. You can change them later.

When installation is finished, from the **Start** menu, open the Anaconda Prompt.

4. *Test your installation*.

## Installing In Silent Mode

NOTE: The following instructions are for Miniconda. For Anaconda, substitute Anaconda for Miniconda in all of the commands.

To run the the Windows installer for Miniconda in *silent mode*, use the /S argument. The following optional arguments are supported:
- /InstallationType=[JustMe|AllUsers]—Default is''JustMe''.

- /AddToPath=[0|1]—Default is 1' - /RegisterPython=[0|1]—Make this the system's default Python. 0 indicates JustMe, which is the default. 1 indicates AllUsers.

- /S—Install in silent mode.

- /D=<installation path>—Destination installation path. Must be the last argument. Do not wrap in quotation marks. Required if you use /S.

All arguments are case-sensitive.

EXAMPLE: The following command installs Miniconda for the current user without registering Python as the system's default:
start /wait "" Miniconda4-latest-Windows-x86_64.exe /InstallationType=JustMe /
˓→RegisterPython=0 /S /D=%UserProfile%\Miniconda3

## Updating Conda

1. Open your Anaconda Prompt from the start menu.

2. Navigate to the anaconda directory. 3. Run conda update conda.

## Uninstalling Conda

1. In the Windows Control Panel, click Add or Remove Program. 2. Select Python X.X (Miniconda), where X.X is your version of Python.

3. Click Remove Program.

NOTE: Removing a program is different in Windows 10.

## Installing On Macos

1. Download the installer:
- Miniconda installer for macOS. - Anaconda installer for macOS.

2. Install:
- Miniconda—In your Terminal window, run:
bash Miniconda3-latest-MacOSX-x86_64.sh
- Anaconda—Double-click the .pkg file.

3. Follow the prompts on the installer screens.

If you are unsure about any setting, accept the defaults. You can change them later.

4. To make the changes take effect, close and then re-open your Terminal window.

5. *Test your installation*.

## Installing In Silent Mode

NOTE: The following instructions are for Miniconda. For Anaconda, substitute Anaconda for Miniconda in all of the commands.

To run the *silent installation* of Miniconda for macOS or Linux, specify the -b and -p arguments of the bash installer.

The following arguments are supported:
- -b—Batch mode with no PATH modifications to ~/.bashrc. Assumes that you agree to the license agreement.

Does not edit the .bashrc or .bash_profile files.

- -p—Installation prefix/path. - -f—Force installation even if prefix -p already exists.

## Example:

wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/
˓→miniconda.sh bash ~/miniconda.sh -b -p $HOME/miniconda export PATH="$HOME/miniconda/bin:$PATH"
NOTE: This sets the PATH only for the current session, not permanently. Trying to use conda when conda is not in your PATH causes errors such as "command not found."
In each new bash session, before using conda, set the PATH and run the activation scripts of your conda packages by running:
source $HOME/miniconda/bin/activate NOTE: Replace $HOME/miniconda/bin/activate with the path to the activate script in your conda installation.

To set the PATH permanently, you can add a line to your .bashrc file. However, this makes it possible to use conda without running the activation scripts of your conda packages, which may produce errors. EXAMPLE:
export PATH="$HOME/miniconda/bin:$PATH"

## Updating Anaconda Or Miniconda

1. Open a Terminal window.

2. Navigate to the anaconda directory. 3. Run conda update conda.

## Uninstalling Anaconda Or Miniconda

1. Open a Terminal window.

2. Remove the entire Miniconda install directory with:
rm -rf ~/miniconda 3. You may also:
4. OPTIONAL: Edit ~/.bash_profile to remove the Miniconda directory from your PATH environment variable.

5. Remove the following hidden file and folders that may have been created in the home directory:
- .condarc file
- .conda directory - .continuum directory By running:
rm -rf ~/.condarc ~/.conda ~/.continuum

## Installing On Linux

1. Download the installer:
- Miniconda installer for Linux.

- Anaconda installer for Linux.

2. In your Terminal window, run:
- Miniconda:
bash Miniconda3-latest-Linux-x86_64.sh
- Anaconda:
bash Anaconda-latest-Linux-x86_64.sh 3. Follow the prompts on the installer screens.

If you are unsure about any setting, accept the defaults. You can change them later.

4. To make the changes take effect, close and then re-open your Terminal window.

5. *Test your installation*.

## Using With Fish Shell

To use conda with fish shell, add the following line in your fish.config file:
source (conda info --root)/etc/fish/conf.d/conda.fish Installing in silent mode See the instructions for *installing in silent mode on macOS*.

Updating Anaconda or Miniconda 1. Open a Terminal window.

2. Run conda update conda.

Uninstalling Anaconda or Miniconda 1. Open a Terminal window. 2. Remove the entire miniconda install directory with:
rm -rf ~/miniconda 3. OPTIONAL: Edit ~/.bash_profile to remove the Miniconda directory from your PATH environment variable.

4. OPTIONAL: Remove the following hidden file and folders that may have been created in the home directory:
- .condarc file
- .conda directory
- .continuum directory By running:
rm -rf ~/.condarc ~/.conda ~/.continuum

## Testing Your Installation

To test your installation, in your Terminal window or Anaconda Prompt, run the command conda list.

For a successful installation, a list of installed packages appears.

# 1.5 **Configuration** 1.5.1 **Using The .Condarc Conda Configuration File**

- *Overview*
- *General configuration*
- *Channel locations (channels)*
- *Allow other channels (allow_other_channels)*
- *Default channels (default_channels)* - *Update conda automatically (auto_update_conda)*
- *Always yes (always_yes)*
- *Show channel URLs (show_channel_urls)*
- *Change command prompt (changeps1)*
- *Add pip as Python dependency (add_pip_as_python_dependency)*
- *Use pip (use_pip)*
- *Configure conda for use behind a proxy server (proxy_servers)*
- *SSL verification (ssl_verify)*
- *Offline mode only (offline)*
- *Advanced configuration*
- *Disallow soft-linking (allow_softlinks)*
- *Set a channel alias (channel_alias)*
- *Always add packages by default (create_default_packages)*
- *Track features (track_features)* - *Disable updating of dependencies (update_dependencies)*
- *Disallow installation of specific packages (disallow)*
- *Add Anaconda.org token to automatically see private packages (add_anaconda_token)* - *Specify environment directories (envs_dirs)*
- *Specify package directories (pkgs_dirs)*
- *Conda build configuration*
- *Specify conda build output root directory (root-dir)*
- *Specify conda build build folder (conda-build 3.16.3+) (output_folder)*
- *Automatically upload conda build packages to Anaconda.org (anaconda_upload)* - *Token to be used for Anaconda.org uploads (conda-build 3.0+) (anaconda_token)*
- *Limit build output verbosity (conda-build 3.0+) (quiet)*
- *Disable filename hashing (conda-build 3.0+) (filename_hashing)*
- *Disable recipe and package verification (conda-build 3.0+) (no_verify)*
- *Disable per-build folder creation (conda-build 3.0+) (set_build_id)*
- *Skip building packages that already exist (conda-build 3.0+) (skip_existing)*

![21_image_0.png](21_image_0.png)

![21_image_1.png](21_image_1.png) - *Omit recipe from package (conda-build 3.0+) (include_recipe)*

![21_image_2.png](21_image_2.png)

![21_image_3.png](21_image_3.png)

- *Disable activation of environments during build/test (conda-build 3.0+) (activate)*

![21_image_4.png](21_image_4.png)

![21_image_5.png](21_image_5.png)

- *Disable long prefix during test (conda-build 3.16.3+) (long_test_prefix)*

![21_image_6.png](21_image_6.png)

![21_image_7.png](21_image_7.png) - *PyPI upload settings (conda-build 3.0+) (pypirc)*
- *PyPI repository to upload to (conda-build 3.0+) (pypi_repository)*
- *Expansion of environment variables* - *Obtaining information from the .condarc file*

## Overview

The conda configuration file, .condarc, is an optional runtime configuration file that allows advanced users to configure various aspects of conda, such as which channels it searches for packages, proxy settings and environment directories.

The .condarc file is not included by default, but it is automatically created in your home directory the first time you run the conda config command. A .condarc file may also be located in the root environment, in which case it overrides any in the home directory.

NOTE: A .condarc file can also be used in an administrator-controlled installation to override the users' configuration. See *Administering a multi-user conda installation*.

The .condarc configuration file follows simple YAML syntax.

The .condarc file can change many parameters, including:
- Where conda looks for packages.

- If and how conda uses a proxy server.

- Where conda lists known environments. - Whether to update the bash prompt with the current activated environment name.

- Whether user-built packages should be uploaded to Anaconda.org.

- Default packages or features to include in new environments.

To create or modify a .condarc file, use the conda config command or use a text editor to create a new file named .condarc and save it to your user home directory or root directory.

EXAMPLE:
conda config --add channels conda-forge You can also download a *sample .condarc file* to edit in your editor and save to your user home directory or root directory.

To set configuration options, edit the .condarc file directly or use the conda config --set command.

EXAMPLE: To set the auto_update_conda option to False, run:
conda config --set auto_update_conda **False**
For a complete list of conda config commands, see the command reference. The same list is available at the Terminal or Anaconda Prompt by running conda config --help.

TIP: Conda supports *tab completion* with external packages instead of internal configuration.

## General Configuration

Channel locations (channels)
Listing channel locations in the .condarc file overrides conda defaults, causing conda to search only the channels listed here, in the order given.

Use defaults to automatically include all default channels. Non-URL channels are interpreted as Anaconda.org user names. You can change this by modifying the channel_alias as described in *Set a channel alias (channel_alias)*.

The default is just defaults.

EXAMPLE:
channels:
- <anaconda_dot_org_username>
- http://some.custom/channel
- file:///some/local/directory - defaults To select channels for a single environment, put a .condarc file in the root directory of that environment (or use the
--env option when using conda config).

EXAMPLE: If you have installed Miniconda with Python 3 in your home directory and the environment is named
"flowers", the path may be:
~/miniconda3/envs/flowers/.condarc

## Allow Other Channels (Allow_Other_Channels)

The system-level .condarc file may specify a set of allowed channels, and it may allow users to install packages from other channels with the boolean flag allow_other_channels. The default is True.

If allow_other_channels is set to False, only those channels explicitly specified in the system .condarc file are allowed:
allow_other_channels: False When allow_other_channels is set to True or not specified, each user has access to the default channels and to any channels that the user specifies in their local .condarc file. When allow_other_channels is set to false, if the user specifies other channels, the other channels are blocked, and the user receives a message reporting that channels are blocked. For more information, see *Example administrator-controlled installation*.

If the system .condarc file specifies a channel_alias, it overrides any channel aliases set in a user's .condarc file.

See *Set a channel alias (channel_alias)*.

## Default Channels (Default_Channels)

Normally the defaults channel points to several channels at the repo.continuum.io repository, but if default_channels is defined, it sets the new list of default channels. This is especially useful for air gap and enterprise installations:
default_channels:
- <anaconda_dot_org_username>
- http://some.custom/channel
- file:///some/local/directory

## Update Conda Automatically (Auto_Update_Conda)

When True, conda updates itself any time a user updates or installs a package in the root environment. When False, conda updates itself only if the user manually issues a conda update command. The default is True.

EXAMPLE:
auto_update_conda: False

## Always Yes (Always_Yes)

Choose the yes option whenever asked to proceed, such as when installing. Same as using the --yes flag at the command line. The default is False.

EXAMPLE:
always_yes: True Show channel URLs (show_channel_urls)
Show channel URLs when displaying what is going to be downloaded and in conda list. The default is False.

EXAMPLE:
show_channel_urls: True Change command prompt (changeps1)
When using activate, change the command prompt from $PS1 to include the activated environment. The default is True.

EXAMPLE:
changeps1: False Add pip as Python dependency (add_pip_as_python_dependency) Add pip, wheel and setuptools as dependencies of Python. This ensures that pip, wheel and setuptools are always installed any time Python is installed. The default is True.

EXAMPLE:
add_pip_as_python_dependency: False

## Use Pip (Use_Pip)

Use pip when listing packages with conda list. This does not affect any conda command or functionality other than the output of the command conda list. The default is True.

EXAMPLE:
use_pip: False Configure conda for use behind a proxy server (proxy_servers) By default, proxy settings are pulled from the HTTP_PROXY and HTTPS_PROXY environment variables or the system. Setting them here overrides that default:
proxy_servers:
http: http://user:pass@corp.com:8080 https: https://user:pass@corp.com:8080 To give a proxy for a specific scheme and host, use the scheme://hostname form for the key. This matches for any request to the given scheme and exact host name:
proxy_servers:
'http://10.20.1.128': 'http://10.10.1.10:5323' If you do not include the user name and password or if authentication fails, conda prompts for a user name and password.

If your password contains special characters, you need escape them as described in Percent-encoding reserved characters , on Wikipedia.

Be careful not to use http when you mean https or https when you mean http.

## Ssl Verification (Ssl_Verify)

If you are behind a proxy that does SSL inspection such as a Cisco IronPort Web Security Appliance (WSA), you may need to use ssl_verify to override the SSL verification settings.

By default this variable is True, which means that SSL verification is used and conda verifies certificates for SSL connections. Setting this variable to False disables the connection's normal security and is not recommended:
ssl_verify: False You can also set ssl_verify to a string path to a certificate, which can be used to verify SSL connections:
ssl_verify: corp.crt

## Offline Mode Only (Offline)

Filters out all channel URLs that do not use the file:// protocol. The default is False.

EXAMPLE:
offline: True

## Advanced Configuration

Disallow soft-linking (allow_softlinks)
When allow_softlinks is True, conda uses hard-links when possible and soft-links—symlinks—when hard-links are not possible, such as when installing on a different file system than the one that the package cache is on.

When allow_softlinks is False, conda still uses hard-links when possible, but when it is not possible, conda copies files. Individual packages can override this option, specifying that certain files should never be soft-linked.

The default is True.

EXAMPLE:
allow_softlinks: False

## Set A Channel Alias (Channel_Alias)

Whenever you use the -c or --channel flag to give conda a channel name that is not a URL, conda prepends the channel_alias to the name that it was given. The default channel_alias is https://conda.anaconda.org/.

EXAMPLE: The command:
conda install --channel asmeurer <package>
is the same as:
conda install --channel https://conda.anaconda.org/asmeurer <package>
You can set channel_alias to your own repository. EXAMPLE: To set channel_alias to your repository at https://yourrepo.com:
channel_alias: https://your.repo/
On Windows, you must include a slash ("/") at the end of the URL:
EXAMPLE: https://your.repo/conda/ When channel_alias set to your repository at https://yourrepo.com:
conda install --channel jsmith <package>
is the same as:
conda install --channel https://yourrepo.com/jsmith <package>

## Always Add Packages By Default (Create_Default_Packages)

When creating new environments, add the specified packages by default. The default packages are installed in every environment you create. You can override this option at the command prompt with the --no-default-packages flag. The default is to not include any packages.

EXAMPLE:
create_default_packages:
- pip
- ipython
- scipy=0.15.0

## Track Features (Track_Features)

Enable certain features to be tracked by default. The default is to not track any features. This is similar to adding mkl to the create_default_packages list.

EXAMPLE:
track_features:
- mkl Disable updating of dependencies (update_dependencies)
By default, conda install updates the given package to the latest version, and installs any dependencies necessary for that package. However if dependencies that satisfy the package's requirements are already installed, conda will not update those packages to the latest version. In this case, if you would prefer that conda update all dependencies to the latest version that is compatible with the environment, set update_dependencies to True:
update_dependencies: False NOTE: Conda still ensures that dependency specifications are satisfied. Thus, some dependencies may still be updated or, conversely, this may prevent packages given at the command line from being updated to their latest versions. You can always specify versions at the command line to force conda to install a given version, such as conda install numpy=1.9.3.

To avoid updating only specific packages in an environment, a better option may be to pin them. For more information, see *Preventing packages from updating (pinning)*.

Disallow installation of specific packages (disallow) Disallow the installation of certain packages. The default is to allow installation of all packages.

EXAMPLE:
disallow:
- anaconda Add Anaconda.org token to automatically see private packages (add_anaconda_token) When the channel alias is Anaconda.org or an Anaconda Server GUI, you can set the system configuration so that users automatically see private packages. Anaconda.org was formerly known as binstar.org. This uses the Anaconda command-line client, which you can install with conda install anaconda-client, to automatically add the token to the channel URLs.

The default is True.

EXAMPLE:
add_anaconda_token: False NOTE: Even when set to True, this setting is enabled only if the Anaconda command-line client is installed and you are logged in with the anaconda login command.

## Specify Environment Directories (Envs_Dirs)

Specify directories in which environments are located. If this key is set, the root prefix envs_dir is not used unless explicitly included. This key also determines where the package caches are located.

For each envs here, envs/pkgs is used as the pkgs cache, except for the standard envs directory in the root directory, for which the normal root_dir/pkgs is used.

EXAMPLE:
envs_dirs:
- ~/my-envs
- /opt/anaconda/envs The CONDA_ENVS_PATH environment variable overwrites this setting:
- For macOS and Linux: CONDA_ENVS_PATH=~/my-envs:/opt/anaconda/envs - For Windows: set CONDA_ENVS_PATH=C:\Users\joe\envs;C:\Anaconda\envs

## Specify Package Directories (Pkgs_Dirs)

Specify directories in which packages are located. If this key is set, the root prefix pkgs_dirs is not used unless explicitly included.

EXAMPLE:
pkgs_dirs:
- /opt/anaconda/pkgs The CONDA_PKGS_DIRS environment variable overwrites this setting:
- For macOS and Linux: CONDA_PKGS_DIRS=/opt/anaconda/pkgs - For Windows: set CONDA_PKGS_DIRS=C:\Anaconda\pkgs

## Conda Build Configuration

Specify conda build output root directory (root-dir)
Build output root directory. You can also set this with the CONDA_BLD_PATH environment variable. The default is
<CONDA_PREFIX>/conda-bld/. If you do not have write permissions to <CONDA_PREFIX>/conda-bld/ , the default is ~/conda-bld/ .

EXAMPLE:
conda-build:
root-dir: ~/conda-builds

## Specify Conda Build Build Folder (Conda-Build 3.16.3+) (Output_Folder)

Folder to dump output package to. Packages are moved here if build or test succeeds. If unset, the output folder corresponds to the same directory as the root build directory (root-dir).

conda-build:
output_folder: conda-bld Automatically upload conda build packages to Anaconda.org (anaconda_upload)
Automatically upload packages built with conda build to Anaconda.org. The default is False.

EXAMPLE:
anaconda_upload: True Token to be used for Anaconda.org uploads (conda-build 3.0+) (anaconda_token)
Tokens are a means of authenticating with anaconda.org without logging in. You can pass your token to condabuild with this condarc setting, or with a CLI argument. This is unset by default. Setting it implicitly enables anaconda_upload.

conda-build:
anaconda_token: gobbledygook Limit build output verbosity (conda-build 3.0+) (quiet)
Conda-build's output verbosity can be reduced with the quiet setting. For more verbosity use the CLI flag --debug.

conda-build:
quiet: true Disable filename hashing (conda-build 3.0+) (filename_hashing)
Conda-build 3 adds hashes to filenames to allow greater customization of dependency versions. If you find this disruptive, you can disable the hashing with the following config entry:
conda-build:
filename_hashing: false NOTE: conda-build does no checking when clobbering packages. If you utilize conda-build 3's build matrices with a build configuration that is not reflected in the build string, packages will be missing due to clobbering. Disable recipe and package verification (conda-build 3.0+) (no_verify) By default, conda-build uses conda-verify to ensure that your recipe and package meet some minimum sanity checks. You can disable these:
conda-build:
no_verify: true

## Disable Per-Build Folder Creation (Conda-Build 3.0+) (Set_Build_Id)

By default, conda-build creates a new folder for each build, named for the package name plus a timestamp. This allows you to do multiple builds at once. If you have issues with long paths, you may need to disable this behavior.

You should first try to change the build output root directory with the root-dir setting described above, but fall back to this as necessary:
conda-build:
set_build_id: false Skip building packages that already exist (conda-build 3.0+) (skip_existing) By default, conda-build builds all recipes that you specify. You can instead skip recipes that are already built. A recipe is skipped if and only if all of its outputs are available on your currently configured channels.

conda-build:
skip_existing: true Omit recipe from package (conda-build 3.0+) (include_recipe)
By default, conda-build includes the recipe that was used to build the package. If this contains sensitive or proprietary information, you can omit the recipe.

conda-build:
include_recipe: false NOTE: If you do not include the recipe, you cannot use conda-build to test the package after the build completes. This means that you cannot split your build and test steps across two distinct CLI commands (conda build --notest recipe and conda build -t recipe). If you need to omit the recipe and split your steps, your only option is to remove the recipe files from the tarball artifacts after your test step. Conda-build does not provide tools for doing that.

## Disable Activation Of Environments During Build/Test (Conda-Build 3.0+) (Activate)

By default, conda-build activates the build and test environments prior to executing the build or test scripts. This adds necessary PATH entries, and also runs any activate.d scripts you may have. If you disable activation, the PATH will still be modified, but the activate.d scripts will not run. This is not recommended, but some people prefer this.

conda-build:
activate: false

## Disable Long Prefix During Test (Conda-Build 3.16.3+) (Long_Test_Prefix)

By default, conda-build uses a long prefix for the test prefix. If you have recipes that fail in long prefixes but would still like to test them in short prefixes, you can disable the long test prefix. This is not recommended.

conda-build:
long_test_prefix: false The default is true.

PyPI upload settings (conda-build 3.0+) (pypirc) Unset by default. If you have wheel outputs in your recipe, conda-build will try to upload them to the PyPI repository specified by the pypi_repository setting using credentials from this file path.

conda-build:
pypirc: ~/.pypirc PyPI repository to upload to (conda-build 3.0+) (pypi_repository) Unset by default. If you have wheel outputs in your recipe, conda-build will try to upload them to this PyPI repository using credentials from the file specified by the pypirc setting.

conda-build:
pypi_repository: pypi

## Expansion Of Environment Variables

Conda expands environment variables in a subset of configuration settings. These are:
- envs_dirs
- pkgs_dirs - ssl_verify
- client_cert
- client_cert_key - proxy_servers
- channels
- custom_channels - custom_multichannels
- default_channels
- migrated_custom_channels - whitelist_channels This allows you to e.g. store the credentials of a private repository in an environment variable, like so:
channels:
- https://${USERNAME}:${PASSWORD}@my.private.conda.channel

## Obtaining Information From The .Condarc File

NOTE: It may be necessary to add the "force" option -f to the following commands.

To get all keys and their values:
conda config --get To get the value of a specific key, such as channels:
conda config --get channels To add a new value, such as http://conda.anaconda.org/mutirri, to a specific key, such as channels:
conda config --add channels http://conda.anaconda.org/mutirri To remove an existing value, such as http://conda.anaconda.org/mutirri from a specific key, such as channels:
conda config --remove channels http://conda.anaconda.org/mutirri To remove a key, such as channels, and all of its values:
conda config --remove-key channels To configure channels and their priority for a single environment, make a .condarc file in the root directory of that environment.

## 1.5.2 **Sample .Condarc File**

\# This is a sample .condarc file. \# It adds the r Anaconda.org channel and enables
\# the show_channel_urls option.

\# channel locations. These override conda defaults, i.e., conda will
\# search *only* the channels listed here, in the order given.

\# Use "defaults" to automatically include all default channels.

\# Non-url channels will be interpreted as Anaconda.org usernames
\# (this can be changed by modifying the channel_alias key; see below).

\# The default is just 'defaults'.

channels:
- r - defaults
\# Show channel URLs when displaying what is going to be downloaded
\# and in 'conda list'. The default is False.

show_channel_urls: True
\# For more information about this file see:
\# https://conda.io/docs/user-guide/configuration/use-condarc.html

## 1.5.3 **Administering A Multi-User Conda Installation**

By default, conda and all packages it installs, including Anaconda, are installed locally with a user-specific configuration. Administrative privileges are not required, and no upstream files or other users are affected by the installation.

You can make conda and any number of packages available to a group of 1 or more users, while preventing these users from installing unwanted packages with conda:
1. Install conda and the allowed packages, if any, in a location that is under administrator control and accessible to users.

2. Create a *.condarc system configuration file* in the root directory of the installation. This system-level configuration file will override any user-level configuration files installed by the user.

Each user accesses the central conda installation, which reads settings from the user .condarc configuration file located in their home directory. The path to the user file is the same as the root environment prefix displayed by conda info, as shown in *User configuration file* below. The user .condarc file is limited by the system .condarc file.

System configuration settings are commonly used in a system .condarc file but may also be used in a user .

condarc file. All user configuration settings may also be used in a system .condarc file.

For information about settings in the .condarc file, see *Using the .condarc conda configuration file*.

## Example Administrator-Controlled Installation

The following example describes how to view the system configuration file, review the settings, compare it to a user's configuration file and determine what happens when the user attempts to access a file from a blocked channel. It then describes how the user must modify their configuration file to access the channels allowed by the administrator.

## System Configuration File

1. The system configuration file must be in the top-level conda installation directory. Check the path where conda is located:
$ which conda /tmp/miniconda/bin/conda 2. View the contents of the .condarc file in the administrator's directory:
cat /tmp/miniconda/.condarc The following administrative .condarc file sets allow_other_channels to False, and it specifies that users may download packages only from the admin channel:
$ cat /tmp/miniconda/.condarc allow_other_channels : false channel_alias: https://conda.anaconda.org/ channels:
- admin NOTE: The admin channel can also be expressed as https://conda.anaconda.org/admin/
Because allow_other_channels is False and the channel defaults are not explicitly specified, users are disallowed from downloading packages from the default channels. You can check this in the next procedure.

## User Configuration File

1. Check the location of the user's conda installation:
$ conda info Current conda install: . . .

channel URLs : http://repo.continuum.io/pkgs/free/osx-64/
http://repo.continuum.io/pkgs/pro/osx-64/
config file : /Users/gergely/.condarc The conda info command shows that conda is using the user's .condarc file, located at /Users/ gergely/.condarc and that the default channels such as repo.continuum.io are listed as channel URLs.

2. View the contents of the administrative .condarc file in the directory that was located in step 1:
$ cat ~/.condarc channels:
- defaults This user's .condarc file specifies only the default channels, but the administrator config file has blocked default channels by specifying that only admin is allowed. If this user attempts to search for a package in the default channels, they get a message telling them what channels are allowed:
$ conda search flask Fetching package metadata:
Error: URL 'http://repo.continuum.io/pkgs/pro/osx-64/' not in allowed channels.

Allowed channels are:
- https://conda.anaconda.org/admin/osx-64/
This error message tells the user to add the admin channel to their configuration file.

3. The user must edit their local .condarc configuration file to access the package through the admin channel:
channels:
- admin The user can now search for packages in the allowed admin channel.

## 1.5.4 **Enabling Tab Completion**

Conda versions up to 4.3 supports tab completion in bash shells via the argcomplete package. Tab completion is deprecated starting with version 4.4. See issue \#415. To enable tab completion:
1. Make sure that argcomplete is installed:
conda install argcomplete 2. Add the following code to your bash profile:
eval "$(register-python-argcomplete conda)"

## 3. Test It:

1. Open a new Terminal window or an Anaconda Prompt.

2. Type: conda ins, and then press the Tab key.

The command completes to:
conda install To get tab completion in zsh, see conda-zsh-completion.

## 1.5.5 **Using Conda On Windows Xp With Or Without A Proxy**

Although Windows XP mainstream support and Extended Support from Microsoft have ended, and Windows XP is no longer one of the target systems supported by Anaconda, some users have had success using Anaconda on Windows XP with the methods described on this page.

Anaconda 2.3.0 is the last version of Python 3-based Anaconda to support Windows XP. Anaconda 2.4 and later have a version of Python 3 built with Visual Studio 2015, which by default does not support Windows XP.

You can install Anaconda 2.3.0 and then update it with conda update conda and conda update --all.

Download Anaconda3-2.3.0-Windows-x86.exe at https://repo.continuum.io/archive/. Install it in any location, such as C:\Anaconda.

## Using A Proxy With Windows Xp

To configure conda for use behind a corporate proxy that uses proxy auto-config (PAC) files and SSL certificates for secure connections:
1. Find a proxy server address from the PAC file:
1. Open Internet Explorer. 2. From the **Tools** menu, select Internet Options, and then click the **Connections** tab. 3. On the **Connections** tab, click the LAN Settings button. 4. In the LAN Settings dialog box, copy the address under the Use automatic configuration script checkbox.

5. Click the Cancel button to close the LAN settings.

6. Click the Cancel button to close the Internet Options.

7. Paste the address into the Internet Explorer address bar, then press the Enter key.

8. In the PAC file that opens, search for return until you find what looks like a proxy IP or DNS address with the port number, which may take some time in a long file.

9. Copy the address and port number.

2. Follow the *.condarc instructions* to create a file named .condarc in your home directory or the installation directory, such as C:\Anaconda\.condarc.

3. Follow the *.condarc proxy server instructions* to add proxy information to the .condarc file.

If you decide to include any passwords, be aware of transformations that can affect special characters. EXAMPLE: This example shows proxy information with passwords:
proxy_servers:
http: http://user:**pass@corp**.com:8080 https: https://user:**pass@corp**.com:8080 ssl_verify: **False**
If you include proxy information without passwords, you will be asked to answer authentication prompts at the command line.

EXAMPLE: This example shows proxy information without passwords:
proxy_servers:
http: http://corp.com:8080 https: https://corp.com:8080 ssl_verify: **False**
Once the proxy is configured, you can run conda update conda and then create and manage environments with the Anaconda Launcher GUI.

Some packages such as flask-login may not be available through conda, so you may need to use pip to install them:
1. To use pip securely over https:
pip install --trusted-host pypi.python.org package-name 2. If the secure https proxy fails, you can force pip to use an insecure http proxy instead:
pip install --index-url=http://pypi.python.org/simple/ --trusted-host pypi.python.

˓→org package-name

## 1.5.6 **Disabling Ssl Verification**

Using conda with SSL is strongly recommended, but it is possible to disable SSL and it may be necessary to disable SSL in certain cases. Some corporate environments use proxy services that use Man-In-The-Middle (MITM) attacks to sniff encrypted traffic. These services can interfere with SSL connections such as those used by conda and pip to download packages from repositories such as PyPI.

If you encounter this interference, you should set up the proxy service's certificates so that the requests package used by conda can recognize and use the certificates.

For cases where this is not possible, conda-build versions 3.0.31 and higher have an option that disables SSL certificate verification and allows this traffic to continue.

conda skeleton pypi can disable SSL verification when pulling packages from a PyPI server over HTTPS.

WARNING: This option causes your computer to download and execute arbitrary code over a connection that it cannot verify as secure. This option is not recommended. Use this option only if necessary. Use this option at your own risk.

To disable SSL verification when using conda skeleton pypi, set the SSL_NO_VERIFY environment variable to either 1 or True (case insensitive).

On *nix systems:
SSL_NO_VERIFY=1 conda skeleton pypi a_package And on Windows systems:
set SSL_NO_VERIFY=1 conda skeleton pypi a_package set SSL_NO_VERIFY=
We recommend that you unset this environment variable immediately after use. If it is not unset, some other tools may recognize it and incorrectly use unverified SSL connections.

Using this option will cause requests to emit warnings to STDERR about insecure settings. If you know that what you're doing is safe, or have been advised by your IT department that what you're doing is safe, you may ignore these warnings.

## 1.6 **Tasks** 1.6.1 **Managing Conda**

- *Verifying that conda is installed*
- *Determining your conda version*
- *Updating conda to the current version*

## Verifying That Conda Is Installed

To verify that conda is installed, in your Terminal window or an Anaconda Prompt, run:
conda --version Conda responds with the version number that you have installed, such as conda 3.11.0.

If you get an error message, make sure of the following:
- You are logged into the same user account that you used to install Anaconda or Miniconda. - You are in a directory that Anaconda or Miniconda can find.

- You have closed and re-opened the Terminal window after installing conda.

## Determining Your Conda Version

In addition to the conda --version command explained above, you can determine what conda version is installed by running one of the following commands in your Terminal window or an Anaconda Prompt:
conda info OR
conda -V

## Updating Conda To The Current Version

To update conda, in your Terminal window or an Anaconda Prompt, run:
conda update conda Conda compares versions and reports what is available to install. It also tells you about other packages that will be automatically updated or changed with the update. If conda reports that a newer version is available, type y to update:
Proceed ([y]/n)? y

## 1.6.2 **Managing Environments**

- *Creating an environment with commands*
- *Creating an environment from an environment.yml file*
- *Cloning an environment*
- *Building identical conda environments*
- *Activating an environment*
- *Deactivating an environment*
- *Determining your current environment*
- *Viewing a list of your environments*
- *Viewing a list of the packages in an environment*
- *Using pip in an environment*
- *Saving environment variables*
- *Sharing an environment*
- *Removing an environment* With conda, you can create, export, list, remove and update environments that have different versions of Python and/or packages installed in them. Switching or moving between environments is called activating the environment. You can also share an environment file.

NOTE: There are many options available for the commands described on this page. For details, see *Command reference*.

## Creating An Environment With Commands

TIP: By default, environments are installed into the envs directory in your conda directory. Run conda create
--help for information on specifying a different path.

Use the Terminal or an Anaconda Prompt for the following steps.

1. To create an environment:
conda create --name myenv NOTE: Replace myenv with the environment name.

2. When conda asks you to proceed, type y:
proceed ([y]/n)?

This creates the myenv environment in /envs/. This environment uses the same version of Python that you are currently using, because you did not specify a version.

To create an environment with a specific version of Python:
conda create -n myenv python=3.4 To create an environment with a specific package:
conda create -n myenv scipy OR:
conda create -n myenv python conda install -n myenv scipy To create an environment with a specific version of a package:
conda create -n myenv scipy=0.15.0 OR:
conda create -n myenv python conda install -n myenv scipy=0.15.0 To create an environment with a specific version of Python and multiple packages:
conda create -n myenv python=3.4 scipy=0.15.0 astroid babel TIP: Install all the programs that you want in this environment at the same time. Installing 1 program at a time can lead to dependency conflicts.

To automatically install pip or another program every time a new environment is created, add the default programs to the *create_default_packages* section of your .condarc configuration file. The default packages are installed every time you create a new environment. If you do not want the default packages installed in a particular environment, use the --no-default-packages flag:
conda create --no-default-packages -n myenv python TIP: You can add much more to the conda create command. For details, run conda create --help.

## Creating An Environment From An Environment.Yml File

Use the Terminal or an Anaconda Prompt for the following steps.

1. Create the environment from the environment.yml file:
conda env create -f environment.yml The first line of the yml file sets the new environment's name. For details see *Creating an environment file manually*.

1. Activate the new environment:
- Windows: activate myenv
- macOS and Linux: source activate myenv NOTE: Replace myenv with the name of the environment.

2. Verify that the new environment was installed correctly:
conda list

## Cloning An Environment

Use the Terminal or an Anaconda Prompt for the following steps.

You can make an exact copy of an environment by creating a clone of it:
conda create --name myclone --clone myenv NOTE: Replace myclone with the name of the new environment. Replace myenv with the name of the existing environment that you want to copy. To verify that the copy was made:
conda info --envs In the environments list that displays, you should see both the source environment and the new copy.

## Building Identical Conda Environments

You can use explicit specification files to build an identical conda environment on the same operating system platform, either on the same machine or on a different machine.

Use the Terminal or an Anaconda Prompt for the following steps.

1. Run conda list --explicit to produce a spec list such as:
\# This file may be used to create an environment using: \# $ conda create --name <env> --file <this file>
\# platform: osx-64
@EXPLICIT
https://repo.continuum.io/pkgs/free/osx-64/mkl-11.3.3-0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/numpy-1.11.1-py35_0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/openssl-1.0.2h-1.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/pip-8.1.2-py35_0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/python-3.5.2-0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/readline-6.2-2.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/setuptools-25.1.6-py35_0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/sqlite-3.13.0-0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/tk-8.5.18-0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/wheel-0.29.0-py35_0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/xz-5.2.2-0.tar.bz2 https://repo.continuum.io/pkgs/free/osx-64/zlib-1.2.8-3.tar.bz2

## 2. To Create This Spec List As A File In The Current Working Directory, Run:

conda list --explicit > spec-file.txt NOTE: You can use spec-file.txt as the filename or replace it with a filename of your choice.

An explicit spec file is not usually cross platform, and therefore has a comment at the top such as \# platform: osx-64 showing the platform where it was created. This platform is the one where this spec file is known to work.

On other platforms, the packages specified might not be available or dependencies might be missing for some of the key packages already in the spec.

To use the spec file to create an identical environment on the same machine or another machine:
conda create --name myenv --file spec-file.txt To use the spec file to install its listed packages into an existing environment:
conda install --name myenv --file spec-file.txt Conda does not check architecture or dependencies when installing from a spec file. To ensure that the packages work correctly, make sure that the file was created from a working environment, and use it on the same architecture, operating system and platform, such as linux-64 or osx-64.

## Activating An Environment

To activate an environment:
- On Windows, in your Anaconda Prompt, run activate myenv
- On macOS and Linux, in your Terminal Window, run source activate myenv Conda prepends the path name myenv onto your system command.

## Deactivating An Environment

To deactivate an environment:
- On Windows, in your Anaconda Prompt, run deactivate - On macOS and Linux, in your Terminal Window, run source deactivate Conda removes the path name myenv from your system command.

TIP: In Windows, it is good practice to deactivate one environment before activating another.

## Determining Your Current Environment

Use the Terminal or an Anaconda Prompt for the following steps.

By default, the active environment—the one you are currently using—is shown in parentheses () or brackets [] at the beginning of your command prompt:
(myenv) $
If you do not see this, run:
conda info --envs In the environments list that displays, your current environment is highlighted with an asterisk (*). By default, the command prompt is set to show the name of the active environment. To disable this option:
conda config --set changeps1 false To re-enable this option:
conda config --set changeps1 true

## Viewing A List Of Your Environments

To see a list of all of your environments, in your Terminal window or an Anaconda Prompt, run:
conda info --envs OR
conda env list A list similar to the following is displayed:
conda environments: myenv /home/username/miniconda/envs/myenv snowflakes /home/username/miniconda/envs/snowflakes bunnies /home/username/miniconda/envs/bunnies

## Viewing A List Of The Packages In An Environment

To see a list of all packages installed in a specific environment:
- If the environment is not activated, in your Terminal window or an Anaconda Prompt, run:
conda list -n myenv
- If the environment is activated, in your Terminal window or an Anaconda Prompt, run:
conda list To see if a specific package is installed in an environment, in your Terminal window or an Anaconda Prompt, run:
conda list -n myenv scipy

## Using Pip In An Environment

To use pip in your environment, in your Terminal window or an Anaconda Prompt, run:
conda install -n myenv pip source activate myenv pip <pip_subcommand>

## Saving Environment Variables

Conda environments can include saved environment variables.

Suppose you want an environment "analytics" to store both a secret key needed to log in to a server and a path to a configuration file. The sections below explain how to write a script named env_vars to do this on *Windows* and macOS or Linux.

This type of script file can be part of a conda package, in which case these environment variables become active when an environment containing that package is activated.

You can name these scripts anything you like. However, multiple packages may create script files, so be sure to use descriptive names that are not used by other packages. One popular option is to give the script a name in the form packagename-scriptname.sh, or on Windows, packagename-scriptname.bat.

## Windows

1. Locate the directory for the conda environment in your Anaconda Prompt by running in the command shell
%CONDA_PREFIX%.

2. Enter that directory and create these subdirectories and files:
cd %CONDA_PREFIX%
mkdir .\etc\conda\activate.d mkdir .\etc\conda\deactivate.d type NUL > .\etc\conda\activate.d\env_vars.bat type NUL > .\etc\conda\deactivate.d\env_vars.bat 3. Edit .\etc\conda\activate.d\env_vars.bat as follows:
set MY_KEY='secret-key-value' set MY_FILE=C:\path\to\my\file 4. Edit .\etc\conda\deactivate.d\env_vars.bat as follows:
set MY_KEY=
set MY_FILE=
When you run activate analytics, the environment variables MY_KEY and MY_FILE are set to the values you wrote into the file. When you run deactivate, those variables are erased.

## Macos And Linux

1. Locate the directory for the conda environment in your Terminal window by running in the terminal echo
$CONDA_PREFIX.
2. Enter that directory and create these subdirectories and files:
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d mkdir -p ./etc/conda/deactivate.d touch ./etc/conda/activate.d/env_vars.sh touch ./etc/conda/deactivate.d/env_vars.sh 3. Edit ./etc/conda/activate.d/env_vars.sh as follows:
\#!/bin/sh export MY_KEY='secret-key-value' export MY_FILE=/path/to/my/file/
4. Edit ./etc/conda/deactivate.d/env_vars.sh as follows:
\#!/bin/sh unset MY_KEY
unset MY_FILE
When you run source activate analytics, the environment variables MY_KEY and MY_FILE are set to the values you wrote into the file. When you run source deactivate, those variables are erased.

## Sharing An Environment

You may want to share your environment with someone else—for example, so they can re-create a test that you have done. To allow them to quickly reproduce your environment, with all of its packages and versions, give them a copy of your environment.yml file.

## Exporting The Environment File

NOTE: If you already have an environment.yml file in your current directory, it will be overwritten during this task.

1. Activate the environment to export:
- On Windows, in your Anaconda Prompt, run activate myenv
- On macOS and Linux, in your Terminal window, run source activate myenv NOTE: Replace myenv with the name of the environment.

2. Export your active environment to a new file:
conda env export > environment.yml NOTE: This file handles both the environment's pip packages and conda packages.

3. Email or copy the exported environment.yml file to the other person.

## Creating An Environment File Manually

You can create an environment file manually to share with others. EXAMPLE: A simple environment file:
name: stats dependencies:
- numpy
- pandas EXAMPLE: A more complex environment file:
name: stats2 channels:
- javascript dependencies:
- python=3.4 *\# or 2.7*
- bokeh=0.9.2
- numpy=1.9.* - nodejs=0.10.*
- flask - pip:
- Flask-Testing You can exclude the default channels by adding nodefaults to the channels list.

channels:
- javascript - nodefaults This is equivalent to passing the --override-channels option to most conda commands.

Adding nodefaults to the channels list in environment.yml is similar to removing defaults from the *channels list* in the .condarc file. However, changing environment.yml affects only one of your conda environments while changing .condarc affects them all.

For details on creating an environment from this environment.yml file, see Creating an environment from an environment.yml file.

## Removing An Environment

To remove an environment, in your Terminal window or an Anaconda Prompt, run:
conda remove --name myenv --all
(You may instead use conda env remove --name myenv.)
To verify that the environment was removed, in your Terminal window or an Anaconda Prompt, run:
conda info --envs The environments list that displays should not show the removed environment.

## 1.6.3 **Managing Channels**

Different channels can have the same package, so conda must handle these channel collisions. There will be no channel collisions if you use only the defaults channel. There will also be no channel collisions if all of the channels you use only contain packages that do not exist in any of the other channels in your list. The way conda resolves these collisions matters only when you have multiple channels in your channel list that host the same package.

## Before Conda 4.1.0

Before conda 4.1.0 was released on June 14, 2016, when two channels hosted packages with the same name, conda installed the package with the highest version number. If there were two packages with the same version number, conda installed the one with the highest build number. Only if both the version numbers and build numbers were identical did the channel ordering make a difference. This approach had 3 problems:
- Build numbers from different channels are not comparable. Channel A could do nightly builds while Channel B does weekly builds, so build 2 from Channel B could be newer than build 4 from Channel A.

- Users could not specify a preferred channel. You might consider Channel B more reliable than Channel A and prefer to get packages from that channel even if the B version is older than the package in Channel A. Conda provided no way to choose that behavior. Only version and build numbers mattered.

- Build numbers conflicted. This is an effect of the other 2 problems. Assume you were happily using package Alpha from Channel A and package Bravo from Channel B. The provider from Channel B then added a version of Alpha with a very high build number. Your conda updates would start installing new versions of Alpha from Channel B whether you wanted that or not. This could cause unintentional problems and a risk of deliberate attacks.

## After Conda 4.1.0

By default, conda now prefers packages from a higher priority channel over any version from a lower priority channel.

Therefore, you can now safely put channels at the bottom of your channel list to provide additional packages that are not in the default channels, and still be confident that these channels will not override the core package set.

Conda collects all of the packages with the same name across all listed channels and processes them as follows:
1. Sorts packages from highest to lowest channel priority.

2. Sorts tied packages—same channel priority—from highest to lowest version number.

3. Sorts still-tied packages—same channel priority and same version—from highest to lowest build number. 4. Installs the first package on the sorted list that satisfies the installation specifications.

To make conda use the old method and install the newest version of a package in any listed channel:
- Add channel_priority: false to your .condarc file.

OR
- Run the equivalent command:
conda config --set channel_priority false Conda then sorts as follows:
1. Sorts the package list from highest to lowest version number.

2. Sorts tied packages from highest to lowest channel priority.

3. Sorts tied packages from highest to lowest build number.

Because build numbers from different channels are not comparable, build number still comes after channel priority.

The following command adds the channel "new_channel" to the top of the channel list, making it the highest priority:
conda config --add channels new_channel Conda now has an equivalent command:
conda config --prepend channels new_channel Conda also now has a command that adds the new channel to the bottom of the channel list, making it the lowest priority:
conda config --append channels new_channel

## 1.6.4 **Creating Custom Channels**

Channels are the path that conda takes to look for packages. The easiest way to use and manage custom channels is to use a private or public repository on Anaconda.org, formerly known as Binstar.org. If you designate your Anaconda.org repository as private, then only you and those you grant access can access your private repository.

If you do not wish to upload your packages to the Internet, you can build a custom repository served either through a web server or locally using a file:// URL.

To create a custom channel:
1. If you have not yet used conda build, install conda build:
conda install conda-build 2. Organize all the packages in subdirectories for the platforms you wish to serve:
channel/
linux-64/
package-1.0-0.tar.bz2 linux-32/
package-1.0-0.tar.bz2 osx-64/
package-1.0-0.tar.bz2 win-64/
package-1.0-0.tar.bz2 win-32/
package-1.0-0.tar.bz2 3. Run conda index on each of the platform subdirectories:
conda index channel/linux-64 channel/osx-64 The conda index command generates a file repodata.json, saved to each repository directory, which conda uses to get the metadata for the packages in the channel.

NOTE: Each time you add or modify a package in the channel, you must rerun conda index for conda to see the update.

4. To test custom channels, serve the custom channel using a web server or using a file:// url to the channel directory. Test by sending a search command to the custom channel.

EXAMPLE: If you want a file in the custom channel location /opt/channel/linux-64/, search for files in that location:
conda search -c file://opt/channel/ --override-channels NOTE: The channel URL does not include the platform, as conda automatically detects and adds the platform.

NOTE: The option --override-channels ensures that conda searches only your specified channel and no other channels, such as default channels or any other channels you may have listed in your .condarc file.

If you have set up your private repository correctly, you get the following output:
Fetching package metadata: . . . .

This is followed by a list of the conda packages found. This verifies that you have set up and indexed your private repository successfully.

## 1.6.5 **Managing Packages**

- *Searching for packages*
- *Installing packages* - *Installing packages from Anaconda.org*
- *Installing non-conda packages*
- *Installing commercial packages*
- *Viewing a list of installed packages*
- *Updating packages*
- *Preventing packages from updating (pinning)*
- *Removing packages* NOTE: There are many options available for the commands described on this page. For details, see *Command reference*.

## Searching For Packages

Use the Terminal or an Anaconda Prompt for the following steps. To see if a specific package such as SciPy is available for installation:
conda search scipy To see if a specific package such as SciPy is available for installation from Anaconda.org:
conda search --override-channels --channel defaults scipy To see if a specific package, such as iminuit, exists in a specific channel, such as http://conda.anaconda.org/mutirri, and is available for installation:
conda search --override-channels --channel http://conda.anaconda.org/mutirri iminuit

## Installing Packages

Use the Terminal or an Anaconda Prompt for the following steps. To install a specific package such as SciPy into an existing environment "myenv":
conda install --name myenv scipy If you do not specify the environment name, which in this example is done by --name myenv, the package installs into the current environment:
conda install scipy To install a specific version of a package such as SciPy:
conda install scipy=0.15.0 To install multiple packages at once, such as SciPy and cURL:
conda install scipy curl NOTE: It is best to install all packages at once, so that all of the dependencies are installed at the same time. To install multiple packages at once and specify the version of the package:
conda install scipy=0.15.0 curl=7.26.0 To install a package for a specific Python version:
conda install scipy=0.15.0 curl=7.26.0 -n py34_env If you want to use a specific Python version, it is best to use an environment with that version. For more information, see *Troubleshooting*.

## Installing Packages From Anaconda.Org

Packages that are not available using conda install can be obtained from Anaconda.org. Formerly Binstar.org, Anaconda.org, is a package management service for both public and private package repositories. Anaconda.org is an Anaconda product, just like Anaconda and Miniconda. To install a package from Anaconda.org:
1. In a browser, go to http://anaconda.org.

2. To find the package named bottleneck, type bottleneck in the top-left box named Search Packages.

3. Find the package that you want and click it to go to the detail page.

The detail page displays the name of the channel. In this example it is the "pandas" channel.

4. Now that you know the channel name, use the conda install command to install the package. In your Terminal window or an Anaconda Prompt, run:
conda install -c pandas bottleneck This command tells conda to install the bottleneck package from the pandas channel on Anaconda.org.

5. To check that the package is installed, in your Terminal window or an Anaconda Prompt, run:
conda list A list of packages appears, including bottleneck.

NOTE: For information on installing packages from multiple channels, see *Managing channels*.

## Installing Non-Conda Packages

If a package is not available from conda or Anaconda.org, you may be able to find and install the package with another package manager like pip.

Pip packages do not have all the features of conda packages, and we recommend first trying to install any package with conda. If the package is unavailable through conda, try installing it with pip. The differences between pip and conda packages cause certain unavoidable limits in compatibility, but conda works hard to be as compatible with pip as possible.

NOTE: Both pip and conda are included in Anaconda and Miniconda, so you do not need to install them separately. NOTE: Conda environments replace virtualenv, so there is no need to activate a virtualenv before using pip.

It is possible to have pip installed outside a conda environment or inside a conda environment.

To gain the benefits of conda integration, be sure to install pip inside the currently active conda environment, and then install packages with that instance of pip. The command conda list shows packages installed this way, with a label showing that they were installed with pip.

You can install pip in the current conda environment with the command conda install pip, as discussed in Using pip in an environment.

If there are instances of pip installed both inside and outside the current conda environment, the instance of pip installed inside the current conda environment is used.

To install a non-conda package:
1. Activate the environment where you want to put the program:
- On Windows, in your Anaconda Prompt, run activate myenv.

- On macOS and Linux,in your Terminal window, run source activate myenv.

2. To use pip to install a program such as See, in your Terminal window or an Anaconda Prompt, run:
pip install see 3. To verify the package was installed, in your Terminal window or an Anaconda Prompt, run:
conda list If the package is not shown, install pip as described in *Using pip in an environment* and try these commands again.

## Installing Commercial Packages

Installing a commercial package such as IOPro is the same as installing any other package. In your Terminal window or an Anaconda Prompt, run:
conda install --name myenv iopro This command installs a free trial of one of Anaconda's commercial packages called IOPro, which can speed up your Python processing. Except for academic use, this free trial expires after 30 days.

## Viewing A List Of Installed Packages

Use the Terminal or an Anaconda Prompt for the following steps.

To list all of the packages in the active environment:
conda list To list all of the packages in a deactivated environment:
conda list -n myenv

## Updating Packages

Use conda update command to check to see if a new update is available. If conda tells you an update is available, you can then choose whether or not to install it. Use the Terminal or an Anaconda Prompt for the following steps.

To update a specific package:
conda update biopython To update Python:
conda update python To update conda itself:
conda update conda NOTE: Conda updates to the highest version in its series, so Python 2.7 updates to the highest available in the 2.x series and 3.6 updates to the highest available in the 3.x series.

To update the Anaconda metapackage:
conda update conda conda update anaconda Regardless of what package you are updating, conda compares versions and then reports what is available to install. If no updates are available, conda reports "All requested packages are already installed."
If a newer version of your package is available and you wish to update it, type y to update:
Proceed ([y]/n)? y

## Preventing Packages From Updating (Pinning)

Pinning a package specification in an environment prevents packages listed in the pinned file from being updated.

In the environment's conda-meta directory, add a file named pinned that includes a list of the packages that you do not want updated.

EXAMPLE: The file below forces NumPy to stay on the 1.7 series, which is any version that starts with 1.7, and forces SciPy to stay at exactly version 0.14.2:
numpy 1.7.*
scipy ==0.14.2 With this pinned file, conda update numpy keeps NumPy at 1.7.1, and conda install scipy=0.15.0 causes an error.

Use the --no-pin flag to override the update restriction on a package. In the Terminal or an Anaconda Prompt, run:
conda update numpy --no-pin Because the pinned specs are included with each conda install, subsequent conda update commands without
--no-pin will revert NumPy back to the 1.7 series.

## Removing Packages

Use the Terminal or an Anaconda Prompt for the following steps. To remove a package such as SciPy in an environment such as myenv:
conda remove -n myenv scipy To remove a package such as SciPy in the current environment:
conda remove scipy To remove multiple packages at once, such as SciPy and cURL:
conda remove scipy curl To confirm that a package has been removed:
conda list

## 1.6.6 **Managing Python**

- *Viewing a list of available Python versions* - *Installing a different version of Python*
- *Using a different version of Python*
- *Updating or upgrading Python* Conda treats Python the same as any other package, so it is easy to manage and update multiple installations.

Anaconda supports Python 2.7, 3.4, 3.5 and 3.6. The default is Python 2.7 or 3.6, depending on which installer you used:
- For the installers "Anaconda" and "Miniconda," the default is 2.7. - For the installers "Anaconda3" or "Miniconda3," the default is 3.6.

## Viewing A List Of Available Python Versions

To list the versions of Python that are available to install, in your Terminal window or an Anaconda Prompt, run:
conda search python This lists all packages whose names contain the text python. To list only the packages whose full name is exactly python, add the --full-name option. In your Terminal window or an Anaconda Prompt, run:
conda search --full-name python

## Installing A Different Version Of Python

To install a different version of Python without overwriting the current version, create a new environment and install the second Python version into it:
1. Create the new environment:
- To create the new environment for Python 3.6, in your Terminal window or an Anaconda Prompt, run:
conda create -n py36 python=3.6 anaconda NOTE: Replace py36 with the name of the environment you want to create. anaconda is the metapackage that includes all of the Python packages comprising the Anaconda distribution. python=3.6 is the package and version you want to install in this new environment. This could be any package, such as numpy=1.7, or *multiple packages*.

- To create the new environment for Python 2.7, in your Terminal window or an Anaconda Prompt, run:
conda create -n py27 python=2.7 anaconda 2. *Activate the new environment*.

3. Verify that the new environment is your *current environment*.

4. To verify that the current environment uses the new Python version, in your Terminal window or an Anaconda Prompt, run:
python --version

## Using A Different Version Of Python

To switch to an environment that has different version of Python, *activate the environment*.

## Updating Or Upgrading Python

Use the Terminal or an Anaconda Prompt for the following steps.

If you are in an environment with Python version 3.4.2, the following command updates Python to the latest version in the 3.4 branch:
conda update python The following command upgrades Python to another branch—3.6—by installing that version of Python:
conda install python=3.6

## 1.6.7 **Using Conda With Travis Ci**

- *The .travis.yml file*
- *Supporting packages that do not have official builds*
- *Building a conda recipe*
- *AppVeyor* If you are already using Travis CI, using conda is a preferable alternative to using apt-get and pip to install packages.

The Debian repos provided by Travis may not include packages for all versions of Python or may not be updated as quickly. Installing such packages with pip may also be undesirable, as this can take a long time, which can consume a large portion of the 50 minutes that Travis allows for each build. Using conda also lets you test the building of conda recipes on Travis.

This page describes how to use conda to test a Python package on Travis CI. However, you can use conda with any language, not just Python.

## The .Travis.Yml File

The following code sample shows how to modify the .travis.yml file to use Miniconda for a project that supports Python 2.7, 3.5 and 3.6:
language: python python:
\# We don't actually use the Travis Python, but this keeps it organized.

- "2.7"
- "3.5"
- "3.6" install:
- sudo apt-get update
\# We do this conditionally because it saves us some downloading if the
\# version is the same.

- if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O
˓→miniconda.sh; else wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O
˓→miniconda.sh; fi
- bash miniconda.sh -b -p $HOME/miniconda - export PATH="$HOME/miniconda/bin:$PATH" - hash -r
- conda config --set always_yes yes --set changeps1 no
- conda update -q conda
\# Useful for debugging any issues with conda
- conda info -a
\# Replace dep1 dep2 ... with your dependencies

![53_image_0.png](53_image_0.png)

- conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION dep1 dep2 ...

- source activate test-environment
- python setup.py install script:
\# Your test script goes here NOTE: For information about the basic configuration for Travis CI, see Building a Python Project.

## Supporting Packages That Do Not Have Official Builds

To support a package that does not have official Anaconda builds:
1. Build the package yourself. 2. Add it to an Anaconda.org channel.

3. Add the following line to the install steps in .travis.yml so that it finds the packages on that channel:
- conda config --add channels your_Anaconda_dot_org_username NOTE: Replace your_Anaconda_dot_org_username with your user name.

## Building A Conda Recipe

If you support official conda packages for your project, you may want to use conda build in Travis, so the building of your recipe is tested as well.

1. Include the conda recipe in the same directory as your source code.

2. In your .travis.yml file, replace the following line:
- python setup.py install with these lines:
- conda build your-conda-recipe
- conda install your-package --use-local

## Appveyor

AppVeyor is a continuous build service for Windows built on Azure and is an alternative to using Travis CI with conda. For an example project building conda packages on AppVeyor, see https://github.com/rmcgibbo/
python-appveyor-conda-example.

## 1.6.8 **Viewing Command-Line Help**

To see a list of supported conda commands, in your Terminal window or an Anaconda Prompt, run:
conda --help OR
conda -h To get help for a specific command, type the command name followed by --help.

EXAMPLE: To see help for the create command, in your Terminal window or an Anaconda Prompt, run:
conda create -h NOTE: You can see the same command help in *Command reference*.

## 1.7 **Cheat Sheet**

See the conda cheat sheet PDF (1 MB) for a single-page summary of the most important information about using conda.

## 1.8 **Troubleshooting**

- *SSL connection errors*
- *Permission denied errors during installation* - *Permission denied errors after using sudo conda command*
- *Already installed error message*
- *Conda reports that a package is installed, but it appears not to be*
- *pkg_resources.DistributionNotFound: conda==3.6.1-6-gb31b0d4-dirty*
- *macOS error "ValueError unknown locale: UTF-8"*
- *AttributeError or missing getproxies*
- *Shell commands open from the wrong location*
- *Programs fail due to invoking conda Python instead of system Python*
- *UnsatisfiableSpecifications error*
- *Package installation fails from a specific channel* - *Conda automatically upgrades to unwanted version*
- *ValidationError: Invalid value for timestamp*
- *Unicode error after installing Python 2*

## 1.8.1 **Ssl Connection Errors** Cause

Installing packages may produce a "connection failed" error if you do not have the certificates for a secure connection to the package repository.

Pip can use the --trusted-host option to indicate that the URL of the repository is trusted:
pip install --trusted-host pypi.org Conda has three similar options.

1. The option --insecure or -k ignores certificate validation errors for all hosts.

Running conda create --help shows:
Networking Options:
-k, --insecure Allow conda to perform "insecure" SSL connections and transfers. Equivalent to setting 'ssl_verify' to
'false'.

2. The configuration option ssl_verify can be set to False.

Running conda config --describe ssl_verify shows:
\# \# ssl_verify (bool, str)
\# \# aliases: verify_ssl \# \# Conda verifies SSL certificates for HTTPS requests, just like a web
\# \# browser. By default, SSL verification is enabled, and conda operations
\# \# will fail if a required url's certificate cannot be verified. Setting
\# \# ssl_verify to False disables certification verification. The value for
\# \# ssl_verify can also be (1) a path to a CA bundle file, or (2) a path \# \# to a directory containing certificates of trusted CA.

\# \#
\# ssl_verify: true Running conda config --set ssl_verify false modifies ~/.condarc and sets the -k flag for all future conda operations performed by that user. Running conda config --help shows other configuration scope options.

When using conda config, the user's conda configuration file at ~/.condarc is used by default. The flag --system will instead write to the system configuration file for all users at <CONDA_BASE_ENV>/
.condarc. The flag --env will instead write to the active conda environment's configuration file at
<PATH_TO_ACTIVE_CONDA_ENV>/.condarc. If --env is used and no environment is active, the user configuration file is used.

3. The configuration option ssl_verify can be used to install new certificates.

Running conda config --describe ssl_verify shows:
\# \# ssl_verify (bool, str)

![56_image_0.png](56_image_0.png)

\# \# aliases: verify_ssl
\# \# Conda verifies SSL certificates for HTTPS requests, just like a web
\# \# browser. By default, SSL verification is enabled, and conda operations
\# \# will fail if a required url's certificate cannot be verified. Setting
\# \# ssl_verify to False disables certification verification. The value for
\# \# ssl_verify can also be (1) a path to a CA bundle file, or (2) a path
\# \# to a directory containing certificates of trusted CA.

\# \# \# ssl_verify: true Your network administrator can give you a certificate bundle for your network's firewall. Then ssl_verify can be set to the path of that certificate authority (CA) bundle, and package installation operations will complete without connection errors.

When using conda config, the user's conda configuration file at ~/.condarc is used by default. The flag --system will instead write to the system configuration file for all users at <CONDA_BASE_ENV>/ .condarc. The flag --env will instead write to the active conda environment's configuration file at <PATH_TO_ACTIVE_CONDA_ENV>/.condarc. If --env is used and no environment is active, the user configuration file is used.

## 1.8.2 **Permission Denied Errors During Installation** Cause

The umask command determines the mask settings that control how file permissions are set for newly created files. If you have a very restrictive umask, such as 077, you get "permission denied" errors.

Set a less restrictive umask before calling conda commands. Conda was intended as a user space tool, but often users need to use it in a global environment. One place this can go awry is with restrictive file permissions. Conda creates links when you install files that have to be read by others on the system. To give yourself full permissions for files and directories, but prevent the group and other users from having access:
1. Before installing, set the umask to 007.

2. Install conda.

3. Return the umask to the original setting:
umask 007 conda install umask 077 For more information on umask, see http://en.wikipedia.org/wiki/Umask.

## 1.8.3 **Permission Denied Errors After Using Sudo Conda Command** Solution

Once you run conda with sudo, you must use sudo forever. We recommend that you NEVER run conda with sudo.

## 1.8.4 **Already Installed Error Message** Cause

If you are trying to fix conda problems without removing the current installation and you try to reinstall Miniconda or Anaconda to fix it, you get an error message that Miniconda or Anaconda is already installed, and you cannot continue.

Install using the –force option.

Download and install the appropriate Miniconda for your operating system from the Miniconda download page using the force option --force or -f:
bash Miniconda3-latest-MacOSX-x86_64.sh -f NOTE: Substitute the appropriate filename and version for your operating system. NOTE: Be sure that you install to the same install location as your existing install so it overwrites the core conda files and does not install a duplicate in a new folder.

## 1.8.5 **Conda Reports That A Package Is Installed, But It Appears Not To Be**

Sometimes conda claims that a package is already installed, but it does not appear to be, for example, a Python package that gives ImportError.

There are several possible causes for this problem, each with its own solution.

## Cause

You are not in the same conda environment as your package.

1. Make sure that you are in the same conda environment as your package. The conda info command tells you what environment is currently active—under default environment.

2. Verify that you are using the Python from the correct environment by running:
import sys print(sys.prefix)

## Cause

For Python packages, you have set the PYTHONPATH or PYTHONHOME variable. These environment variables cause Python to load files from locations other than the standard ones. Conda works best when these environment variables are not set, as their typical use cases are obviated by conda environments and a common issue is that they cause Python to pick up the wrong versions or broken versions of a library.

For Python packages, make sure you have not set the PYTHONPATH or PYTHONHOME variables. The command conda info -a displays the values of these environment variables.

- To unset these environment variables temporarily for the current Terminal session, run unset PYTHONPATH.

- To unset them permanently, check for lines in the files:
- If you use bash—~/.bashrc, ~/.bash_profile, ~/.profile.

- If you use zsh—*~/.zshrc'*.

- If you use PowerShell on Windows, the file output by $PROFILE .

## Cause

You have site-specific directories or, for Python, you have so-called site-specific files. These are typically located in
~/.local on Linux and macOS. For a full description of the locations of site-specific packages, see PEP 370. As with PYTHONPATH, Python may try importing packages from this directory, which can cause issues.

For Python packages, remove site-specific directories and site-specific files.

## Cause

For C libraries, the following environment variables have been set:
- macOS—DYLD_LIBRARY_PATH.

- Linux—LD_LIBRARY_PATH.

These act similarly to PYTHONPATH for Python. If they are set, they can cause libraries to be loaded from locations other than the conda environment. Conda environments obviate most use cases for these variables. The command conda info -a shows what these are set to.

Unset DYLD_LIBRARY_PATH or LD_LIBRARY_PATH.

## Cause

Occasionally, an installed package becomes corrupted. Conda works by unpacking the packages in the pkgs directory and then hard-linking them to the environment. Sometimes these get corrupted, breaking all environments that use them, and also any additional environments, since the same files are hard-linked each time.

Run the command conda install -f to unarchive the package again and relink it. It also does an md5 verification on the package. Usually if this is different, it is because your channels have changed and there is a different package with the same name, version, and build number.

NOTE: This breaks the links to any other environments that already had this package installed, so you have to reinstall it there, too. It also means that running conda install -f a lot can use up a lot of disk space if you have a lot of environments.

NOTE: The -f flag to conda install (--force) implies --no-deps, so conda install -f package does not reinstall any of the dependencies of package.

## 1.8.6 **Pkg_Resources.Distributionnotfound: Conda==3.6.1-6-Gb31B0D4-Dirty** Cause

The local version of conda needs updating.

Force reinstall conda. A useful way to work off the development version of conda is to run python setup.py develop on a checkout of the conda git repository. However, if you are not regularly running git pull, it is a good idea to un-develop, as you will otherwise not get any regular updates to conda. The normal way to do this is to run python setup.py develop -u.

However, this command does not replace the conda script itself. With other packages, this is not an issue, as you can just reinstall them with conda, but conda cannot be used if conda is installed.

The fix is to use the ./bin/conda executable in the conda git repository to force reinstall conda, that is, run ./
bin/conda install -f conda. You can then verify with conda info that you have the latest version of conda, and not a git checkout—the version should not include any hashes.

## 1.8.7 **Macos Error "Valueerror Unknown Locale: Utf-8"** Cause

This is a bug in the macOS Terminal app that shows up only in certain locales. Locales are country-language combinations.

1. Open Terminal in /Applications/Utilities 2. Clear the Set locale environment variables on startup checkbox.

![54_image_0.png](54_image_0.png)

This sets your LANG environment variable to be empty. This may cause Terminal use to incorrect settings for your locale. The locale command in Terminal tells you what settings are used.

To use the correct language, add a line to your bash profile, which is typically ~/.profile:
export LANG=your-lang NOTE: Replace your-lang with the correct locale specifier for your language. The command locale -a displays all the specifiers. For example, the language code for US English is en_US.

UTF-8. The locale affects what translations are used when they are available and also how dates, currencies and decimals are formatted.

## 1.8.8 **Attributeerror Or Missing Getproxies**

When running a command such as conda update ipython, you may get an AttributeError:
'module' object has no attribute 'getproxies'.

## Cause

This can be caused by an old version of requests or by having the PYTHONPATH environment variable set.

Update requests and be sure PYTHONPATH is not set:
1. Run conda info -a to show the requests version and various environment variables such as PYTHONPATH.

2. Update the requests version with pip install -U requests. 3. Clear PYTHONPATH:
- On Windows, clear it the environment variable settings.

- On macOS and Linux, clear it by removing it from the bash profile and restarting the shell.

## 1.8.9 **Shell Commands Open From The Wrong Location**

When you run a command within a conda environment, conda does not access the correct package executable.

## Cause

In both bash and zsh, when you enter a command, the shell searches the paths in PATH one by one until it finds the command. The shell then caches the location, which is called hashing in shell terminology. When you run command again, the shell does not have to search the PATH again.

The problem is that before you installed the program, you ran a command which loaded and hashed another version of that program in some other location on the PATH, such as /usr/bin. Then you installed the program using conda install, but the shell still had the old instance hashed.

Reactivate the environment or run hash -r (in bash) or rehash (in zsh).

When you run source activate, conda automatically runs hash -r in bash and rehash in zsh to clear the hashed commands, so conda finds things in the new path on the PATH. But there is no way to do this when conda install is run because the command must be run inside the shell itself, meaning either you have to run the command yourself or use source a file that contains the command.

This is a relatively rare problem, since this happens only in the following circumstances:
1. You activate an environment or use the root environment, and then run a command from somewhere else.

2. Then you conda install a program, and then try to run the program again without running activate or deactivate.

The command type command_name always tells you exactly what is being run. This is better than which command_name, which ignores hashed commands and searches the PATH directly. The hash is reset by source activate, or by hash -r in bash or rehash in zsh.

## 1.8.10 **Programs Fail Due To Invoking Conda Python Instead Of System Python** Cause

After installing Anaconda or Miniconda, programs that run python switch from invoking the system Python to invoking the Python in the root conda environment. If these programs rely on the system Python to have certain configurations or dependencies that are not in the root conda environment Python, the programs may crash. For example, some users of the Cinnamon desktop environment on Linux Mint have reported these crashes.

Edit your .bash_profile and .bashrc files so that the conda binary directory, such as ~/miniconda3/bin, is no longer added to the PATH environment variable. You can still run conda activate and deactivate by using their full path names, such as ~/miniconda3/bin/conda.

You may also create a folder with symbolic links to conda, activate and deactivate, and then edit your . bash_profile or .bashrc file to add this folder to your PATH. If you do this, running python will invoke the system Python, but running conda commands, source activate MyEnv, source activate root, or source deactivate will work normally.

After running source activate to activate any environment, including after running source activate root, running python will invoke the Python in the active conda environment.

## 1.8.11 **Unsatisfiablespecifications Error** Cause

Some conda package installation specifications are impossible to satisfy. For example, conda create -n tmp python=3 wxpython=3 produces an "Unsatisfiable Specifications" error because wxPython 3 depends on Python 2.7, so the specification to install Python 3 conflicts with the specification to install wxPython 3. When an unsatisfiable request is made to conda, conda shows a message such as this one:
The following specifications were found to be in conflict:
- python 3*
- wxpython 3* -> python 2.7* Use "conda info <package>" to see the dependencies for each package.

This indicates that the specification to install wxpython 3 depends on installing Python 2.7, which conflicts with the specification to install python 3.

Use "conda info wxpython" or "conda info wxpython=3" to show information about this package and its dependencies:
wxpython 3.0 py27_0 -------------------
file name : wxpython-3.0-py27_0.tar.bz2 name : wxpython version : 3.0 build number: 0 build string: py27_0 channel : defaults size : 34.1 MB
date : 2014-01-10 fn : wxpython-3.0-py27_0.tar.bz2 license_family: Other md5 : adc6285edfd29a28224c410a39d4bdad priority : 2 schannel : defaults url : https://repo.continuum.io/pkgs/free/osx-64/wxpython-3.0-py27_0.tar.bz2 dependencies:
python 2.7* python.app By examining the dependencies of each package, you should be able to determine why the installation request produced a conflict and modify the request so it can be satisfied without conflicts. In this example, you could install wxPython with Python 2.7:
conda create -n tmp python=2.7 wxpython=3

## 1.8.12 **Package Installation Fails From A Specific Channel** Cause

Sometimes it is necessary to install a specific version from a specific channel because that version is not available from the default channel.

The following example describes the problem in detail and its solution.

Suppose you have a specific need to install the Python cx_freeze module with Python 3.4. A first step is to create a Python 3.4 environment:
conda create -n py34 python=3.4 Using this environment you should first attempt:
conda install -n py34 cx_freeze However, when you do this you get the following error:
Using Anaconda Cloud api site https://api.anaconda.org Fetching package metadata .........

Solving package specifications: .

Error: Package missing in current osx-64 channels:
- cx_freeze You can search for packages on anaconda.org **with**
anaconda search -t conda cx_freeze The message indicates that cx_freeze cannot be found in the default package channels. However, there may be a community-created version available and you can search for it by running the following command:
$ anaconda search -t conda cx_freeze Using Anaconda Cloud api site https://api.anaconda.org Run 'anaconda show <USER/PACKAGE>' to get more details: Packages:
Name | Version | Package Types | Platforms ------------------------- | ------ | --------------- | ---------------
inso/cx_freeze | 4.3.3 | conda | linux-64 pyzo/cx_freeze | 4.3.3 | conda | linux-64, win-32, win-
˓→64, linux-32, osx-64
: http://cx-freeze.sourceforge.net/
silg2/cx_freeze | 4.3.4 | conda | linux-64
: create standalone executables from Python
˓→scripts takluyver/cx_freeze | 4.3.3 | conda | linux-64 Found 4 packages In this example, there are 4 different places that you could try to get the package. None of them are officially supported or endorsed by Anaconda, but members of the conda community have provided many valuable packages. If you want to go with public opinion, then the web interface provides more information:
Notice that the pyzo organization has by far the most downloads, so you might choose to use their package. If so, you can add their organization's channel by specifying it on the command line:
$ conda create -c pyzo -n cxfreeze_py34 cx_freeze python=3.4 Using Anaconda Cloud api site https://api.anaconda.org Fetching package metadata: ..........

Solving package specifications: .........

Package plan for installation in environment /Users/ijstokes/anaconda/envs/cxfreeze_
˓→py34:
(continues on next page)

| () ®  https://beta.anaconda.org/search?q=cx_freeze   | FO   | 2         |             |
|------------------------------------------------------|------|-----------|-------------|
|                                                      | Docs | Contact - | 91 ijstokes |
| NACONDA C                                            |      |           |             |
| cx_freeze                                            | q    |           |             |

cx_freeze

| - Filters                                         |               |                             |           |
|---------------------------------------------------|---------------|-----------------------------|-----------|
| Type: All ^                                       | Access: All ^ | Platform: All ×             |           |
| � Favorites                                       | - Downloads   | ♦ Package (owner / package) | Platforms |
| linux-32                                          |               |                             |           |
| linux-64                                          |               |                             |           |
| O pyzo / cx_freeze 4.3.3                          |               |                             |           |
| o                                                 | 1976          | osx-64                      |           |
| http://cx-freeze.sourceforge.net/                 | couda         | win-32                      |           |
| win-64                                            |               |                             |           |
| e pypi / cx_Freeze 43.3                           |               |                             |           |
| 96                                                |               |                             |           |
| o                                                 | source        |                             |           |
| create standalone executables from Python scripts | pypi          |                             |           |
| O inso / cx_freeze 4.3.3                          |               |                             |           |
| o                                                 | 14            | linux-64                    |           |
| conda                                             |               |                             |           |
| O silg2 / cx_freeze 43.4                          |               |                             |           |
| o                                                 | 1             | linux-64                    |           |
| create standalone executables from Python scripts |               |                             |           |
| 3 takluyver / cx_freeze 43.3                      |               |                             |           |
| o                                                 | o             | linux-64                    |           |
| conda                                             |               |                             |           |

 Next »
« Previous showing 1 - 5 of 5

| (continued from previous page)                                                                                                                                                                                                                  |        |        |        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|--------|--------|
| The following packages will be downloaded: package |                                                                                                                                                                                            | build  |        |        |
| ---------------------------|----------------- cx_freeze-4.3.3 | py34_4                                                                                                                                                                          | 1.8 MB |        |        |
| setuptools-20.7.0                                                                                                                                                                                                                               | |      | py34_0 | 459 KB |
| ------------------------------------------------------------ Total: 2.3 MB                                                                                                                                                                      |        |        |        |
| The following NEW packages will be INSTALLED: cx_freeze: 4.3.3-py34_4 openssl: 1.0.2h-0 pip: 8.1.1-py34_1 python: 3.4.4-0 readline: 6.2-2 setuptools: 20.7.0-py34_0 sqlite: 3.9.2-0 tk: 8.5.18-0 wheel: 0.29.0-py34_0 xz: 5.0.5-1 zlib: 1.2.8-0 |        |        |        |

Now you have a software environment sandbox created with Python 3.4 and cx_freeze.

## 1.8.13 **Conda Automatically Upgrades To Unwanted Version**

When making a python package for an app, you create an environment for the app from a file req.txt that sets a certain version, such as python=2.7.9. However, when you conda install your package, it automatically upgrades to a later version, such as 2.7.10.

## Cause

If you make a conda package for the app using conda build, you can set dependencies with specific version numbers.

The requirements lines that say - python could be - python ==2.7.9 instead. It is important to have 1 space before the == operator and no space after.

Exercise caution when coding version requirements.

## 1.8.14 **Validationerror: Invalid Value For Timestamp** Cause

This happens when certain packages are installed with conda 4.3.28, and then conda is downgraded to 4.3.27 or earlier.

See https://github.com/conda/conda/issues/6096.

## 1.8.15 **Unicode Error After Installing Python 2**

Example: UnicodeDecodeError: 'ascii' codec can't decode byte 0xd3 in position 1: ordinal not in range(128)

## Cause

Python 2 is incapable of handling unicode properly, especially on Windows. In this case, if any character in your PATH env. var contains anything that is not ASCII then you see this exception.

Remove all non-ASCII from PATH or switch to Python 3.

CHAPTER 2 Conda Commands

## 2.1 Conda Create

Create a new conda environment from a list of specified packages. To use the created environment, use 'source activate envname' look in that directory first. This command requires either the -n NAME or -p PREFIX option.

Options:

| usage: conda create [-h] [--clone ENV] [-n ENVIRONMENT | -p PATH] [-c CHANNEL] [--use-local] [--override-channels] [--strict-channel-priority] [--no-channel-priority] [--no-deps | --only-deps] [--no-pin] [--copy] [-C] [-k] [--offline] [-d] [--json] [-q] [-v] [-y] [--download-only] [--show-channel-urls] [--file FILE] [--no-default-packages] [package_spec [package_spec ...]]   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 2.1.1 **Positional Arguments**

package_spec Packages to install or update in the conda environment.

## 2.1.2 **Named Arguments**

--clone Path to (or name of) existing local environment.

--file Read package versions from the given file. Repeated file specifications can be passed (e.g. –file=file1 –file=file2).

## 2.1.3 **Target Environment Specification**

-n, --name Name of environment.

-p, --prefix Full path to environment location (i.e. prefix).

## 2.1.4 **Channel Customization**

-c, --channel **Additional channel to search for packages. These are URLs searched in the order**
they are given (including file:// for local directories). Then, the defaults or channels from .condarc are searched (unless –override-channels is given). You can use 'defaults' to get the default packages for conda. You can also use any name and the .condarc channel_alias value will be prepended. The default channel_alias is http://conda.anaconda.org/.

--use-local Use locally built packages. Identical to '-c local'. --override-channels Do not search default or .condarc channels. Requires –channel.

## 2.1.5 **Solver Mode Modifiers**

--strict-channel-priority Packages in lower priority channels are not considered if a package with the same name appears in a higher priority channel.

--no-channel-priority Package version takes precedence over channel priority. Overrides the value given by *conda config –show channel_priority*.

--no-deps Do not install, update, remove, or change dependencies. This WILL lead to broken environments and inconsistent behavior. Use at your own risk.

--no-pin Ignore pinned file.

--only-deps Only install dependencies.

--no-default-packages Ignore create_default_packages in the .condarc file.

## 2.1.6 **Package Linking And Install-Time Options**

--copy Install all packages using copies instead of hard- or soft-linking.

## 2.1.7 **Networking Options**

-C, --use-index-cache Use cache of channel index files, even if it has expired. -k, --insecure Allow conda to perform "insecure" SSL connections and transfers. Equivalent to setting 'ssl_verify' to 'false'.

--offline Offline mode. Don't connect to the Internet.

## 2.1.8 **Output, Prompt, And Flow Control Options**

-d, --dry-run Only display what would have been done. -q, --quiet Do not display progress bar.

| --json        | Report all output as json. Suitable for using conda programmatically.              |
|---------------|------------------------------------------------------------------------------------|
| -v, --verbose | Can be used multiple times. Once for INFO, twice for DEBUG, three times for TRACE. |

-y, --yes Do not ask for confirmation.

--download-only Solve an environment and ensure package caches are populated, but exit prior to

unlinking and linking packages into the prefix.

--show-channel-urls Show channel urls. Overrides the value given by *conda config –show* show_channel_urls.

Examples:
conda create -n myenv sqlite

## 2.2 Conda Install

Installs a list of packages into a specified conda environment.

This command accepts a list of package specifications (e.g, bitarray=0.8) and installs a set of packages consistent with those specifications and compatible with the underlying environment. If full compatibility cannot be assured, an error is reported and the environment is not changed.

Conda attempts to install the newest versions of the requested packages. To accomplish this, it may update some packages that are already installed, or install additional packages. To prevent existing packages from updating, use the –freeze-installed option. This may force conda to install older versions of the requested packages, and it does not prevent additional dependency packages from being installed.

If you wish to skip dependency checking altogether, use the '–no-deps' option. This may result in an environment with incompatible packages, so this option must be used with great caution.

conda can also be called with a list of explicit conda package filenames (e.g. ./lxml-3.2.0-py27_0.tar.bz2).

Using conda in this mode implies the –no-deps option, and should likewise be used with great caution.

Explicit filenames and package specifications cannot be mixed in a single command.

## Options: 2.2.1 **Positional Arguments**

package_spec Packages to install or update in the conda environment.

| usage: conda install [-h] [--revision REVISION] [-n ENVIRONMENT | -p PATH] [-c CHANNEL] [--use-local] [--override-channels] [--strict-channel-priority] [--no-channel-priority] [--no-deps | --only-deps] [--no-pin] [--copy] [-C] [-k] [--offline] [-d] [--json] [-q] [-v] [-y] [--download-only] [--show-channel-urls] [--file FILE] [--prune] [--force-reinstall] [--freeze-installed | --update-deps | -S | --update-all] [-m] [--clobber] [package_spec [package_spec ...]]   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 2.2.2 **Named Arguments**

--revision Revert to the specified REVISION.

| --file   | Read package versions from the given file. Repeated file specifications can be passed (e.g. –file=file1 –file=file2).   |
|----------|-------------------------------------------------------------------------------------------------------------------------|

## 2.2.3 **Target Environment Specification**

-n, --name Name of environment.

-p, --prefix Full path to environment location (i.e. prefix).

## 2.2.4 **Channel Customization**

-c, --channel **Additional channel to search for packages. These are URLs searched in the order**
they are given (including file:// for local directories). Then, the defaults or channels from .condarc are searched (unless –override-channels is given). You can use 'defaults' to get the default packages for conda. You can also use any name and the .condarc channel_alias value will be prepended. The default channel_alias is http://conda.anaconda.org/.

--use-local Use locally built packages. Identical to '-c local'.

--override-channels Do not search default or .condarc channels. Requires –channel.

## 2.2.5 **Solver Mode Modifiers**

--strict-channel-priority Packages in lower priority channels are not considered if a package with the same name appears in a higher priority channel.

## 2.2.6 **Package Linking And Install-Time Options**

-m, --mkdir Create the environment directory if necessary.

| same name appears in a higher priority channel.                                                                           |                                                                                                                                                          |
|---------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| --no-channel-priority                                                                                                     | Package version takes precedence over channel priority. Overrides the value                                                                              |
| given by conda config –show channel_priority.                                                                             |                                                                                                                                                          |
| --no-deps                                                                                                                 | Do not install, update, remove, or change dependencies. This WILL lead to broken environments and inconsistent behavior. Use at your own risk.                                                                                                                                                          |
| --only-deps                                                                                                               | Only install dependencies.                                                                                                                               |
| --no-pin                                                                                                                  | Ignore pinned file.                                                                                                                                      |
| --prune                                                                                                                   | Remove packages that have previously been brought into the environment to satisfy dependencies of user-requested packages, but are no longer needed.                                                                                                                                                          |
| --force-reinstall                                                                                                         | Ensure that any user-requested package for the current operation is uninstalled and reinstalled, even if that package already exists in the environment. |
| --freeze-installed, --no-update-deps                                                                                      | Do not update or change already-installed dependencies.                                                                                                  |
| --update-deps                                                                                                             | Update dependencies.                                                                                                                                     |
| -S, --satisfied-skip-solve                                                                                                | Exit early and do not run the solver if the requested specs are satisfied. Also                                                                          |
| skips aggressive updates as configured by 'aggressive_update_packages'. Similar to the default behavior of 'pip install'. |                                                                                                                                                          |
| --update-all, --all                                                                                                       | Update all installed packages in the environment.                                                                                                        |

| --copy    | Install all packages using copies instead of hard- or soft-linking.                        |
|-----------|--------------------------------------------------------------------------------------------|
| --clobber | Allow clobbering of overlapping file paths within packages, and suppress related warnings. |

## 2.2.7 **Networking Options**

-C, --use-index-cache Use cache of channel index files, even if it has expired.

-k, --insecure Allow conda to perform "insecure" SSL connections and transfers. Equivalent to setting 'ssl_verify' to 'false'.

--offline Offline mode. Don't connect to the Internet.

## 2.2.8 **Output, Prompt, And Flow Control Options**

| -d, --dry-run       | Only display what would have been done.                                                                                         |         |       |           |     |       |       |    |              |       |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------|---------|-------|-----------|-----|-------|-------|----|--------------|-------|
| --json              | Report all output as json. Suitable for using conda programmatically.                                                           |         |       |           |     |       |       |    |              |       |
| -q, --quiet         | Do not display progress bar.                                                                                                    |         |       |           |     |       |       |    |              |       |
| -v, --verbose       | Can be used multiple times. Once for INFO, twice for DEBUG, three times for TRACE.                                              |         |       |           |     |       |       |    |              |       |
| -y, --yes           | Do not ask for confirmation.                                                                                                    |         |       |           |     |       |       |    |              |       |
| --download-only     | Solve an environment and ensure package caches are populated, but exit prior to unlinking and linking packages into the prefix. |         |       |           |     |       |       |    |              |       |
| --show-channel-urls | Show                                                                                                                            | channel | urls. | Overrides | the | value | given | by | conda config | –show |
| show_channel_urls.  |                                                                                                                                 |         |       |           |     |       |       |    |              |       |

Examples:
conda install -n myenv scipy

## 2.3 Conda Update

Updates conda packages to the latest compatible version.

This command accepts a list of package names and updates them to the latest versions that are compatible with all other packages in the environment.

Conda attempts to install the newest versions of the requested packages. To accomplish this, it may update some packages that are already installed, or install additional packages. To prevent existing packages from updating, use the –no-update-deps option. This may force conda to install older versions of the requested packages, and it does not prevent additional dependency packages from being installed.

## Options:

| usage: conda update [-h] [-n ENVIRONMENT | -p PATH] [-c CHANNEL] [--use-local] [--override-channels] [--strict-channel-priority] [--no-channel-priority] [--no-deps | --only-deps] [--no-pin] [--copy] [-C] [-k] [--offline] [-d] [--json] [-q] [-v] [-y] [--download-only] [--show-channel-urls] [--file FILE] [--prune] [--force-reinstall] [--freeze-installed | --update-deps | -S | --update-all] [--clobber] [package_spec [package_spec ...]]   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 2.3.1 **Positional Arguments**

package_spec Packages to install or update in the conda environment.

## 2.3.2 **Named Arguments**

--file Read package versions from the given file. Repeated file specifications can be passed (e.g. –file=file1 –file=file2).

## 2.3.3 **Target Environment Specification**

-n, --name Name of environment.

-p, --prefix Full path to environment location (i.e. prefix).

## 2.3.4 **Channel Customization**

-c, --channel **Additional channel to search for packages. These are URLs searched in the order**
they are given (including file:// for local directories). Then, the defaults or channels from .condarc are searched (unless –override-channels is given).

You can use 'defaults' to get the default packages for conda. You can also use any name and the .condarc channel_alias value will be prepended. The default channel_alias is http://conda.anaconda.org/.

--use-local Use locally built packages. Identical to '-c local'. --override-channels Do not search default or .condarc channels. Requires –channel.

## 2.3.5 **Solver Mode Modifiers**

--strict-channel-priority Packages in lower priority channels are not considered if a package with the same name appears in a higher priority channel.

--update-deps Update dependencies. -S, --satisfied-skip-solve Exit early and do not run the solver if the requested specs are satisfied. Also skips aggressive updates as configured by 'aggressive_update_packages'. Similar to the default behavior of 'pip install'.

| --no-channel-priority                         | Package version takes precedence over channel priority. Overrides the value                                                                              |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| given by conda config –show channel_priority. |                                                                                                                                                          |
| --no-deps                                     | Do not install, update, remove, or change dependencies. This WILL lead to broken environments and inconsistent behavior. Use at your own risk.                                                                                                                                                          |
| --only-deps                                   | Only install dependencies.                                                                                                                               |
| --no-pin                                      | Ignore pinned file.                                                                                                                                      |
| --prune                                       | Remove packages that have previously been brought into the environment to satisfy dependencies of user-requested packages, but are no longer needed.                                                                                                                                                          |
| --force-reinstall                             | Ensure that any user-requested package for the current operation is uninstalled and reinstalled, even if that package already exists in the environment. |
| --freeze-installed, --no-update-deps          | Do not update or change already-installed dependencies.                                                                                                  |

--update-all, --all Update all installed packages in the environment.

## 2.3.6 **Package Linking And Install-Time Options**

--copy Install all packages using copies instead of hard- or soft-linking.

--clobber Allow clobbering of overlapping file paths within packages, and suppress related warnings.

## 2.3.7 **Networking Options**

-C, --use-index-cache Use cache of channel index files, even if it has expired. -k, --insecure Allow conda to perform "insecure" SSL connections and transfers. Equivalent to setting 'ssl_verify' to 'false'.

--offline Offline mode. Don't connect to the Internet.

## 2.3.8 **Output, Prompt, And Flow Control Options**

--show-channel-urls Show channel urls. Overrides the value given by *conda config –show* show_channel_urls.

| -d, --dry-run   | Only display what would have been done.                                                                                         |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------|
| --json          | Report all output as json. Suitable for using conda programmatically.                                                           |
| -q, --quiet     | Do not display progress bar.                                                                                                    |
| -v, --verbose   | Can be used multiple times. Once for INFO, twice for DEBUG, three times for TRACE.                                              |
| -y, --yes       | Do not ask for confirmation.                                                                                                    |
| --download-only | Solve an environment and ensure package caches are populated, but exit prior to unlinking and linking packages into the prefix. |

Examples:
conda update -n myenv scipy

## 2.4 Conda Remove

Remove a list of packages from a specified conda environment.

This command will also remove any package that depends on any of the specified packages as well—unless a replacement can be found without that dependency. If you wish to skip this dependency checking and remove just the requested packages, add the '–force' option. Note however that this may result in a broken environment, so use this with caution.

## Options:

| usage: conda remove [-h] [-n ENVIRONMENT | -p PATH] [-c CHANNEL] [--use-local] [--override-channels] [--all] [--features] [--force-remove] [--no-pin] [--prune] [-C] [-k] (continues on next page)   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| (continued from previous page)                                             |
|----------------------------------------------------------------------------|
| [--offline] [-d] [--json] [-q] [-v] [-y] [package_name [package_name ...]] |

## 2.4.1 **Positional Arguments**

package_name Package names to remove from the environment.

## 2.4.2 **Target Environment Specification**

-n, --name Name of environment. -p, --prefix Full path to environment location (i.e. prefix).

## 2.4.3 **Channel Customization**

-c, --channel **Additional channel to search for packages. These are URLs searched in the order**
they are given (including file:// for local directories). Then, the defaults or channels from .condarc are searched (unless –override-channels is given).

You can use 'defaults' to get the default packages for conda. You can also use any name and the .condarc channel_alias value will be prepended. The default channel_alias is http://conda.anaconda.org/.

--use-local Use locally built packages. Identical to '-c local'.

--override-channels Do not search default or .condarc channels. Requires –channel.

## 2.4.4 **Solver Mode Modifiers** 2.4.5 **Networking Options**

-C, --use-index-cache Use cache of channel index files, even if it has expired.

-k, --insecure Allow conda to perform "insecure" SSL connections and transfers. Equivalent to setting 'ssl_verify' to 'false'.

| --all                                                                                     | Remove all packages, i.e., the entire environment.                       |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| --features                                                                                | Remove features (instead of packages).                                   |
| --force-remove, --force                                                                   | Forces removal of a package without removing packages that depend on it. |
| Using this option will usually leave your environment in a broken and inconsistent state. |                                                                          |
| --no-pin                                                                                  | Ignore pinned file.                                                      |
| --prune                                                                                   | Remove packages that have previously been brought into the environment to satisfy dependencies of user-requested packages, but are no longer needed.                                                                          |

--offline Offline mode. Don't connect to the Internet.

| -d, --dry-run   | Only display what would have been done.                                            |
|-----------------|------------------------------------------------------------------------------------|
| --json          | Report all output as json. Suitable for using conda programmatically.              |
| -q, --quiet     | Do not display progress bar.                                                       |
| -v, --verbose   | Can be used multiple times. Once for INFO, twice for DEBUG, three times for TRACE. |

## 2.4.6 **Output, Prompt, And Flow Control Options**

-y, --yes Do not ask for confirmation.

Examples:
conda remove -n myenv scipy

## 2.5 Conda Info

Display information about current conda install. Options:

| usage: conda info [-h] [--json] [-v] [-q] [-a] [--base] [-e] [-s] [--unsafe-channels]   |
|-----------------------------------------------------------------------------------------|

## 2.5.1 **Named Arguments**

-a, --all Show all information.

--base Display base environment path.

-e, --envs List all known conda environments.

-s, --system List environment variables.

--unsafe-channels Display list of channels with tokens exposed.

## 2.5.2 **Output, Prompt, And Flow Control Options**

--json Report all output as json. Suitable for using conda programmatically. -v, --verbose Use once for info, twice for debug, three times for trace.

-q, --quiet Do not display progress bar.

## 2.6 Conda Search

Search for packages and display associated information. The input is a MatchSpec, a query language for conda packages. See examples below.

Options:

| usage: conda search [-h] [--envs] [-i] [--subdir SUBDIR] [-c CHANNEL] [--use-local] [--override-channels] [-C] [-k] [--offline] [--json] [-v] [-q]   |
|------------------------------------------------------------------------------------------------------------------------------------------------------|

## 2.6.1 **Named Arguments**

--envs Search all of the current user's environments. If run as Administrator (on Windows) or UID 0 (on unix), search all known environments on the system.

-i, --info Provide detailed information about each package.

--subdir, --platform Search the given subdir. Should be formatted like 'osx-64', 'linux-32', 'win-64',
and so on. The default is to search the current platform.

## 2.6.2 **Channel Customization**

-c, --channel **Additional channel to search for packages. These are URLs searched in the order**
they are given (including file:// for local directories). Then, the defaults or channels from .condarc are searched (unless –override-channels is given).

You can use 'defaults' to get the default packages for conda. You can also use any name and the .condarc channel_alias value will be prepended. The default channel_alias is http://conda.anaconda.org/.

--use-local Use locally built packages. Identical to '-c local'.

--override-channels Do not search default or .condarc channels. Requires –channel.

## 2.6.3 **Networking Options**

-C, --use-index-cache Use cache of channel index files, even if it has expired. -k, --insecure Allow conda to perform "insecure" SSL connections and transfers. Equivalent to setting 'ssl_verify' to 'false'.

--offline Offline mode. Don't connect to the Internet.

## 2.6.4 **Output, Prompt, And Flow Control Options**

--json Report all output as json. Suitable for using conda programmatically.

-v, --verbose Use once for info, twice for debug, three times for trace.

-q, --quiet Do not display progress bar.

Examples: Search for a specific package named 'scikit-learn':
conda search scikit-learn Search for packages containing 'scikit' in the package name:
conda search *scikit* Note that your shell may expand '*' before handing the command over to conda. Therefore it is sometimes necessary to use single or double quotes around the query.

conda search '*scikit' conda search "*scikit*"
Search for packages for 64-bit Linux (by default, packages for your current platform are shown):
conda search numpy[subdir=linux-64]
Search for a specific version of a package:
conda search 'numpy>=1.12' Search for a package on a specific channel conda search conda-forge::numpy conda search 'numpy[channel=conda-forge, subdir=osx-64]'

## 2.7 Conda Config

Modify configuration values in .condarc. This is modeled after the git config command. Writes to the user .condarc file (/home/docs/.condarc) by default.

Options:

| usage: conda config [-h] [--json] [-v] [-q] [--system | --env | --file FILE] [--show [SHOW [SHOW ...]] | --show-sources | --validate | --describe [DESCRIBE [DESCRIBE ...]] | --write-default] [--get [KEY [KEY ...]] | --append KEY VALUE | --prepend KEY VALUE | --set KEY VALUE | --remove KEY VALUE | --remove-key KEY | --stdin]   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 2.7.1 **Output, Prompt, And Flow Control Options**

| --json        | Report all output as json. Suitable for using conda programmatically.   |
|---------------|-------------------------------------------------------------------------|
| -v, --verbose | Use once for info, twice for debug, three times for trace.              |

-q, --quiet Do not display progress bar.

## 2.7.2 **Config File Location Selection**

Without one of these flags, the user config file at '/home/docs/.condarc' is used.

--file Write to the given file.

| --system   | Write to the system .condarc file at '/home/docs/checkouts/readthedocs.org/user_builds/continuumioconda/envs/4.6.0/.condarc'.                                                                                                                                                                  |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --env      | Write to the active conda environment .condarc file (<no active environment>). If no environment is active, write to the user config file (/home/docs/.condarc). |

## 2.7.3 **Config Subcommands**

| --show          | Display configuration values as calculated and compiled. If no arguments given, show information for all configuration values.   |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------|
| --show-sources  | Display all identified configuration sources.                                                                                    |
| --validate      | Validate all configuration sources.                                                                                              |
| --describe      | Describe given configuration parameters. If no arguments given, show information for all configuration parameters.                                                                                                                                  |
| --write-default | Write the default configuration to a file. Equivalent to conda config –describe > ~/.condarc.                                    |

## 2.7.4 **Config Modifiers**

--get Get a configuration value.

| --append         | Add one configuration value to the end of a list key.                                  |
|------------------|----------------------------------------------------------------------------------------|
| --prepend, --add | Add one configuration value to the beginning of a list key.                            |
| --remove         | Remove a configuration value from a list key. This removes all instances of the value. |
| --stdin          | Apply configuration information given in yaml format piped through stdin.              |

--set Set a boolean or string key

--remove-key Remove a configuration key (and all its values).

See *conda config –describe* or https://conda.io/docs/config.html for details on all the options that can go in .condarc.

Examples: Display all configuration values as calculated and compiled:
conda config –show Display all identified configuration sources:
conda config –show-sources Describe all available configuration options:
conda config –describe Add the conda-canary channel:
conda config –add channels conda-canary Set the output verbosity to level 3 (highest) for the current activate environment:
conda config –set verbosity 3 –env Add the 'conda-forge' channel as a backup to 'defaults':
conda config –append channels conda-forge

## 2.8 Conda List

List linked packages in a conda environment.

Options:

## 2.8.1 **Positional Arguments**

| usage: conda list [-h] [-n ENVIRONMENT | -p PATH] [--json] [-v] [-q] [--show-channel-urls] [-c] [-f] [--explicit] [--md5] [-e] [-r] [--no-pip] [regex]   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|

regex List only packages matching this regular expression.

## 2.8.2 **Named Arguments**

| --show-channel-urls   | Show                                                                                              | channel   | urls.   | Overrides   | the   | value   | given   | by   | conda config   | –show   |
|-----------------------|---------------------------------------------------------------------------------------------------|-----------|---------|-------------|-------|---------|---------|------|----------------|---------|
| show_channel_urls.    |                                                                                                   |           |         |             |       |         |         |      |                |         |
| -c, --canonical       | Output canonical names of packages only. Implies –no-pip.                                         |           |         |             |       |         |         |      |                |         |
| -f, --full-name       | Only search for full names, i.e., ^<regex>$.                                                      |           |         |             |       |         |         |      |                |         |
| --explicit            | List explicitly all installed conda packaged with URL (output may be used by conda create –file). |           |         |             |       |         |         |      |                |         |
| --md5                 | Add MD5 hashsum when using –explicit                                                              |           |         |             |       |         |         |      |                |         |
| -e, --export          | Output requirement string only (output may be used by conda create –file).                        |           |         |             |       |         |         |      |                |         |
| -r, --revisions       | List the revision history and exit.                                                               |           |         |             |       |         |         |      |                |         |
| --no-pip              | Do not include pip-only installed packages.                                                       |           |         |             |       |         |         |      |                |         |

## 2.8.3 **Target Environment Specification**

-n, --name Name of environment.

-p, --prefix Full path to environment location (i.e. prefix).

## 2.8.4 **Output, Prompt, And Flow Control Options**

--json Report all output as json. Suitable for using conda programmatically. -v, --verbose Use once for info, twice for debug, three times for trace.

-q, --quiet Do not display progress bar.

Examples: List all packages in the current environment:
conda list List all packages installed into the environment 'myenv':
conda list -n myenv Save packages for future use:
conda list –export > package-list.txt Reinstall packages from an export file:
conda create -n myenv –file package-list.txt

## 2.9 Conda Clean

Remove unused packages and caches.

Options:

| usage: conda clean [-h] [-a] [-i] [-l] [-p] [-t] [-f] [-d] [--json] [-q] [-v] [-y]   |
|--------------------------------------------------------------------------------------|

## 2.9.1 **Removal Targets**

| -a, --all             | Remove index cache, lock files, unused cache packages, and tarballs.                                                                                                                        |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -i, --index-cache     | Remove index cache.                                                                                                                                                                         |
| -l, --lock            | Remove all conda lock files.                                                                                                                                                                |
| -p, --packages        | Remove unused packages from writable package caches. WARNING: This does not check for packages installed using symlinks back to the package cache.                                          |
| -t, --tarballs        | Remove cached package tarballs.                                                                                                                                                             |
| -f, --force-pkgs-dirs | Remove all writable package caches. This option is not included with the –all flag. WARNING: This will break environments with packages installed using symlinks back to the package cache. |

## 2.9.2 **Output, Prompt, And Flow Control Options**

-d, --dry-run Only display what would have been done.

| --json        | Report all output as json. Suitable for using conda programmatically.              |
|---------------|------------------------------------------------------------------------------------|
| -v, --verbose | Can be used multiple times. Once for INFO, twice for DEBUG, three times for TRACE. |

-q, --quiet Do not display progress bar.

-y, --yes Do not ask for confirmation.

Examples:
conda clean –tarballs

## 2.10 Conda Package

Low-level conda package utility. (EXPERIMENTAL)
Options:

## 2.10.1 **Named Arguments**

| usage: conda package [-h] [-n ENVIRONMENT | -p PATH] [-w PATH [PATH ...]] [-r] [-u] [--pkg-name PKG_NAME] [--pkg-version PKG_VERSION] [--pkg-build PKG_BUILD]   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|

-r, --reset Remove all untracked files and exit.

| -w, --which   | Given some PATH print which conda package the file came from.   |
|---------------|-----------------------------------------------------------------|

-u, --untracked Display all untracked files and exit.

--pkg-name Package name of the created package. --pkg-version Package version of the created package.

--pkg-build Package build number of the created package.

## 2.10.2 **Target Environment Specification**

| -n, --name   | Name of environment.                             |
|--------------|--------------------------------------------------|
| -p, --prefix | Full path to environment location (i.e. prefix). |
