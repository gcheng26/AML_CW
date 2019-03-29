def get_labels():
    """

    Returns: A list with the category of each file in wav_filenames.csv

    """
    filenames = [line.rstrip('\n') for line in open('wav_filenames.csv')]
    patient_diag = [line.rstrip('\n') for line in open('patient_diagnosis.csv')]

    diagnose = []
    for i in range(len(patient_diag)):
        diag = patient_diag[i].rsplit(',')
        diagnose.append(diag)

    # For each patient id in filenames, append the diagnosis into a list.
    labels = []
    for name in filenames:
        i = 0
        while diagnose[i][0] != name[:3]:
            if i == 500:
                print('Alarm! i is 500.')
            i = i+1
        labels.append(diagnose[i][1])
    return labels
