local SpatialMedianPooling, parent = torch.class('nn.SpatialMedianPooling', 'nn.Module')

function SpatialMedianPooling:__init(kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.indices = torch.Tensor()
end

function SpatialMedianPooling:updateOutput(input)
   input.nn.SpatialMedianPooling_updateOutput(self, input)
   return self.output
end

function SpatialMedianPooling:updateGradInput(input, gradOutput)
   input.nn.SpatialMedianPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialMedianPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end