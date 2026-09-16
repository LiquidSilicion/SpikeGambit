// ============================================================================
// Module: conv_layer
// Description: 3x3 convolution layer with LIF neuron activation
//              Processes input spikes and generates output spikes
// ============================================================================

module conv_layer #(
    parameter IN_CHANNELS = 12,
    parameter OUT_CHANNELS = 32,
    parameter BOARD_SIZE = 8,
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH = 16,
    parameter WEIGHT_ADDR_WIDTH = 12
)(
    input  wire clk,
    input  wire rst_n,
    
    // Input spike interface
    input  wire                    in_spike,
    input  wire [$clog2(IN_CHANNELS)-1:0] in_channel,
    input  wire [$clog2(BOARD_SIZE)-1:0] in_x,
    input  wire [$clog2(BOARD_SIZE)-1:0] in_y,
    
    // Weight memory interface
    output wire [WEIGHT_ADDR_WIDTH-1:0] weight_addr,
    input  wire signed [DATA_WIDTH-1:0] weight_data,
    
    // Output spike interface
    output reg                     out_spike,
    output reg [$clog2(OUT_CHANNELS)-1:0] out_channel,
    output reg [$clog2(BOARD_SIZE)-1:0] out_x,
    output reg [$clog2(BOARD_SIZE)-1:0] out_y,
    
    // Control
    output reg                     busy
);

    // State machine
    localparam IDLE = 3'd0;
    localparam LOAD_WEIGHTS = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam UPDATE_NEURON = 3'd3;
    localparam CHECK_FIRE = 3'd4;
    
    reg [2:0] state;
    
    // Kernel position counters
    reg [1:0] kx, ky;  // 0, 1, 2
    reg [$clog2(OUT_CHANNELS)-1:0] oc;  // Output channel
    
    // Accumulator for convolution
    reg signed [DATA_WIDTH-1:0] accumulator;
    
    // LIF neurons for output feature map
    // Each output channel at each spatial location has its own LIF neuron
    reg signed [DATA_WIDTH-1:0] membrane [0:OUT_CHANNELS-1][0:BOARD_SIZE-1][0:BOARD_SIZE-1];
    
    // Threshold (Q8.8 format, 0.5 = 128)
    localparam signed [DATA_WIDTH-1:0] THRESHOLD = 16'sh0080;
    
    // Weight address calculation
    // Address = (out_channel * in_channels * kernel_size^2) + 
    //           (in_channel * kernel_size^2) + 
    //           (ky * kernel_size + kx)
    wire [WEIGHT_ADDR_WIDTH-1:0] calc_weight_addr;
    assign calc_weight_addr = (oc * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE) +
                              (in_channel * KERNEL_SIZE * KERNEL_SIZE) +
                              (ky * KERNEL_SIZE + kx);
    
    assign weight_addr = calc_weight_addr;
    
    // Initialize membrane potentials
    integer i, j, k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < OUT_CHANNELS; i = i + 1) begin
                for (j = 0; j < BOARD_SIZE; j = j + 1) begin
                    for (k = 0; k < BOARD_SIZE; k = k + 1) begin
                        membrane[i][j][k] <= 16'sd0;
                    end
                end
            end
        end
    end
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            out_spike <= 1'b0;
            busy <= 1'b0;
            kx <= 2'd0;
            ky <= 2'd0;
            oc <= 0;
            accumulator <= 16'sd0;
        end else begin
            case (state)
                IDLE: begin
                    out_spike <= 1'b0;
                    busy <= 1'b0;
                    if (in_spike) begin
                        state <= LOAD_WEIGHTS;
                        busy <= 1'b1;
                        kx <= 2'd0;
                        ky <= 2'd0;
                        oc <= 0;
                        accumulator <= 16'sd0;
                    end
                end
                
                LOAD_WEIGHTS: begin
                    // Weight data is available from weight_memory
                    state <= COMPUTE;
                end
                
                COMPUTE: begin
                    // Accumulate: acc += weight (since in_spike is 1)
                    accumulator <= accumulator + weight_data;
                    
                    // Move to next kernel position
                    if (kx < 2'd2) begin
                        kx <= kx + 2'd1;
                        state <= LOAD_WEIGHTS;
                    end else if (ky < 2'd2) begin
                        kx <= 2'd0;
                        ky <= ky + 2'd1;
                        state <= LOAD_WEIGHTS;
                    end else begin
                        // Finished 3x3 kernel for this output channel
                        state <= UPDATE_NEURON;
                    end
                end
                
                UPDATE_NEURON: begin
                    // Update membrane potential with leak
                    // V_new = V_old + acc - (V_old >> 3)  [leak factor 0.875]
                    membrane[oc][in_x][in_y] <= membrane[oc][in_x][in_y] + 
                                                accumulator - 
                                                (membrane[oc][in_x][in_y] >>> 3);
                    
                    // Move to next output channel
                    if (oc < OUT_CHANNELS - 1) begin
                        oc <= oc + 1;
                        kx <= 2'd0;
                        ky <= 2'd0;
                        accumulator <= 16'sd0;
                        state <= LOAD_WEIGHTS;
                    end else begin
                        // All output channels processed
                        state <= CHECK_FIRE;
                    end
                end
                
                CHECK_FIRE: begin
                    // Check if any neuron fired
                    out_spike <= 1'b0;
                    for (i = 0; i < OUT_CHANNELS; i = i + 1) begin
                        if (membrane[i][in_x][in_y] >= THRESHOLD) begin
                            out_spike <= 1'b1;
                            out_channel <= i[$clog2(OUT_CHANNELS)-1:0];
                            out_x <= in_x;
                            out_y <= in_y;
                            membrane[i][in_x][in_y] <= 16'sd0;  // Reset after spike
                        end
                    end
                    state <= IDLE;
                    busy <= 1'b0;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule