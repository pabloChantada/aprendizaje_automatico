using FileIO
using Images
using JLD2
using Pkg

using Flux
using Flux.Losses

using DelimitedFiles


function newClassCascadeNetwork(numInputs::Int, numOutputs::Int) 

    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputs, numOutputs, σ));
    
    else
        ann = Chain(ann..., Dense(numInputs, numOutputs, identity))
        ann = Chain(ann..., softmax);
    end;
    
    return ann;
end;
    
