using FileIO
using Images
using JLD2
using Pkg

using Flux
using Flux.Losses

using DelimitedFiles


function randomImages(numImages::Int, resolution::Int)
    matrix = randn(numImages, 1, resolution, resolution)
    result = matrix .> 0 
    return result
end;



function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})

    labels = unique(labelArray)
    C, H, W = size(imageArray)[2:end]
    num_labels = lenght(labels)

    outputArray = similar(imageArray, typeof(imageArray[1]), num_labels, C, H, W)

    for (indexLabel, digit) in num_labels
        
        digit_images = imageArray[labelArray .== digit, :, :, :]
        averaged_image = dropdims(mean(imageArray[labelArray.==labels[indexLabel], 1, :, :], dims=1), dims=1)
        outputArray[indexLabel, :, :, :] .= averaged_image

    end
    
    return (outputArray, labels)
end;


    
        
        