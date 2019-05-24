function fig = newFigure(varargin)
%NEWFIGURE Summary of this function goes here
%   Detailed explanation goes here

    size_x = getAdditionalParam( 'size_x', varargin, 900 );
    size_y = getAdditionalParam( 'size_y', varargin, 680 );

    fig = figure('position', [0 0 size_x size_y]);
end

