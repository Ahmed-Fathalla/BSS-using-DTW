import shutil, os
def listdirs(rootdir):
    for it in os.scandir(rootdir):
    
        ######################################
        # Check if directory
        ######################################
        if it.is_dir():
            listdirs(it)
            
        if it.path.endswith('_OUTPUT.csv') or it.path.endswith('.pkl') or \
           it.path.endswith(' _ Errors.txt'):# in it:
#             print(it.path)
            
            #######################################################
            #    remove files
            #######################################################
            os.remove()    

            ##################################################
            #    remove files
            #######################################################
            shutil.rmtree(it.path)  

p = 'Class_1 20 runs/Calss_1 20 run/'
listdirs(p)