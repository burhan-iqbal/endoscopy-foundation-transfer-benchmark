# CE-NBI Data Audit

- Dataset root: `/Users/aqib/Documents/Project/dissertation_project/data/interim/ce_nbi`
- Excel metadata: `/Users/aqib/Documents/Project/dissertation_project/data/interim/ce_nbi/Patients_List_Updated_Final.xlsx`
- Images found: **11144** (expected 11144)
- Patients found: **210** (expected 210)
- Image-level class counts: {'benign': 7657, 'malignant': 3487}
- Patient-level class counts: {'benign': 150, 'malignant': 60}
- Patients with conflicting binary labels: **0**

## Histopathology breakdown (image counts)

lesion_type       histopathology    n
     benign          Amyloidosis   87
     benign                 Cyst  407
     benign            Granuloma   92
     benign           Hemangioma   75
     benign       Hyperkeratosis  675
     benign          Hyperplasia  180
     benign         Inflammation    9
     benign  Low_grade_dysplasia 1428
     benign          Namboo_node   93
     benign               Nodule   26
     benign       Papillomatosis 1103
     benign                Polyp  821
     benign        Reinkes_edema 2661
  malignant    Carcinoma_in_situ  542
  malignant High_grade_dysplasia 1039
  malignant                  SCC 1906

## Notes

- Raw zip is never modified; interim extract is used for analysis.
- Binary target is benign (0) vs malignant (1).
- Patient IDs are taken from `PatientXXX` folders (normalized to `PXXX`), with Excel leukoplakia/histopathology joined when available.
