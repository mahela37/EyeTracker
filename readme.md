# git clone: clones a git repository to your local 
- eg: git clone https://github.com/mahela37/EyeTracker.git

# Submitting code:

1. git add -A
    - adds all files to commit
2. git commit -A    
    - do a commit with a meaningful commit message
3. git push     
    - pushes files to cloud

# Retrieving latest code: git pull

# Python libraries used in the project: 


# Recommended Python IDE: PyCharm
- https://www.jetbrains.com/pycharm/
- Makes the GIT workflow much easier    

#Mahela -- Used Libraries
- CV2
    - Make sure Windows Media Pack is installed if getting DLL import errors
- dlib
    - This one was difficult. Ideally "pip install dlib" should do it. I got a lot of errors,and the following was what worked for me:
    1. pip install cmake
    2. Install cmake on windows
    3. Install Visual Studio Community (Choose the C# and C++ options so that CMake is included)
    4. pip install dlib
     