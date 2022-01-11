ls /nfs/data/main/M32/RegistrationData/Data/$1/legacy/Transformation_OUTPUT_CH3/$1_img/*.jp2 | xargs -n 1 basename > /nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/$1/listF.txt
A=$(ls /nfs/data/main/M32/RegistrationData/Data/$1/legacy/Transformation_OUTPUT_CH3/$1_img/*.jp2|wc -l)
q00=0
q01=$(($A/4))
q10=$(($q01+1))
q11=$(($A/2))
q20=$(($q11+1))
q21=$((3*$A/4))
q30=$(($q21+1))
q31=$(($A-1))

echo (($q00 $q01 $q10 $q11 $q20 $q21 $q30 $q31))

# python test_frcnn_full_DK_May_2021_GPU0.py DK39 $q00 $q01
# python test_frcnn_full_DK_May_2021_GPU0.py DK39 $q10 $q11
# python test_frcnn_full_DK_May_2021_GPU1.py DK39 $q20 $q21
# python test_frcnn_full_DK_May_2021_GPU1.py DK39 $q30 $q31

Error: no such file "0"
Error: no such file "117"
Error: no such file "118"
Error: no such file "234"
Error: no such file "235"
Error: no such file "351"
Error: no such file "352"
Error: no such file "468"



