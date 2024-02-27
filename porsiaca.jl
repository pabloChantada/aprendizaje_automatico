    #=
    if vp + fn == 0
        sensitivity = 1
    else
        sensitivity = vp / (fn + vp)
    end

    if vn + fp == 0
        specificity = 1
    else
        specificity = vn / (vn + fp)
    end

    if vp + fp == 0
        positive_predictive_value = 1
    else
        positive_predictive_value = vp / (vp + fp)
    end
     
    if vn + fn == 0
        negative_predictive_value = 1
    else
        negative_predictive_value = vn / (vn + fn)
    end
    # como haces la media armonica sin vecotores ??¿?¿?
    if sensitivity + positive_predictive_value == 0
        f_score = 0
    else
        f_score = (positive_predictive_value * sensitivity) / (positive_predictive_value + sensitivity)
    end
    return matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix 
end;
=# 