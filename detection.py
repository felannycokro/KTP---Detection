from do_ktp_detection import do_detection

def detection(img_path):
    try:
        predictions = do_detection(img_path)

        # labeling image status as Yes or No
        try :
            # set minimum threshold of probability
            if predictions[0]['probability'] > 0.5:
                return "Yes"
            else:
                return "No"
        except:
            return "No"

    except Exception as e:
        return e
