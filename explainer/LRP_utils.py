import numpy as np
import matplotlib.pypplot as plt
class LRP_utils(object):
    def __init__(self):

    def get_data(self):
        domain_pad_test = np.load("domain_pad_test.npy")
        domain_pad_resampled = np.load("domain_pad_resampled.npy")
        X_train = np.load("X_train.npy")
        domain_pad_test = np.load("domain_pad_test.npy")
        target_train = np.load("target_train.npy")
        target_test = np.load("target_test.npy")

        return domain_pad_resampled, domain_pad_test, target_train, target_test, X_train

    def html_heatmap (self, words, scores, cmap_name="bwr"):
        colormap  = plt.get_cmap(cmap_name)

        #assert len(words)==len(scores)
        max_s     = max(scores)
        min_s     = min(scores)

        output_text = ""

        for idx, w in enumerate(words):
            score       = self.rescale_score_by_abs(scores[idx], max_s, min_s)
            output_text = output_text + self.span_word(w, score, colormap) + " "

        return output_text + "\n"

    def rescale_score_by_abs (self, score, max_score, min_score):

        # CASE 1: positive AND negative scores occur
        if max_score>0 and min_score<0:

            if max_score >= abs(min_score):   # deepest color is positive
                if score>=0:
                    return 0.5 + 0.5*(score/max_score)
                else:
                    return 0.5 - 0.5*(abs(score)/max_score)

            else:                             # deepest color is negative
                if score>=0:
                    return 0.5 + 0.5*(score/abs(min_score))
                else:
                    return 0.5 - 0.5*(score/min_score)

                # CASE 2: ONLY positive scores occur
        elif max_score>0 and min_score>=0:
            if max_score == min_score:
                return 1.0
            else:
                return 0.5 + 0.5*(score/max_score)

        # CASE 3: ONLY negative scores occur
        elif max_score<=0 and min_score<0:
            if max_score == min_score:
                return 0.0
            else:
                return 0.5 - 0.5*(score/min_score)

    #################### heatmap format

    def getRGB (self, c_tuple):
        return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

    def span_word (self,word, score, colormap):
        return "<span style=\"background-color:"+ self.getRGB(colormap(score))+"\">"+word+"</span>"