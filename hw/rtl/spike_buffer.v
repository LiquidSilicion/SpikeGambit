// ============================================================================
// Module: spike_buffer
// Description: FIFO buffer for spike routing between layers
// ============================================================================

module spike_buffer #(
    parameter DATA_WIDTH = 16,
    parameter BUFFER_DEPTH = 256
)(
    input  wire clk,
    input  wire rst_n,
    
    // Write interface
    input  wire write_en,
    input  wire spike,
    input  wire [4:0] channel,
    input  wire [2:0] x_coord,
    input  wire [2:0] y_coord,
    
    // Read interface
    output reg read_en,
    output reg spike_out,
    output reg [4:0] channel_out,
    output reg [2:0] x_out,
    output reg [2:0] y_out,
    
    // Status
    output wire empty,
    output wire full
);

    // Buffer storage
    reg spike_buf [0:BUFFER_DEPTH-1];
    reg [4:0] channel_buf [0:BUFFER_DEPTH-1];
    reg [2:0] x_buf [0:BUFFER_DEPTH-1];
    reg [2:0] y_buf [0:BUFFER_DEPTH-1];
    
    // Pointers
    reg [$clog2(BUFFER_DEPTH)-1:0] write_ptr;
    reg [$clog2(BUFFER_DEPTH)-1:0] read_ptr;
    reg [$clog2(BUFFER_DEPTH):0] count;
    
    assign empty = (count == 0);
    assign full = (count == BUFFER_DEPTH);
    
    // Write logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
        end else if (write_en && !full) begin
            spike_buf[write_ptr] <= spike;
            channel_buf[write_ptr] <= channel;
            x_buf[write_ptr] <= x_coord;
            y_buf[write_ptr] <= y_coord;
            write_ptr <= write_ptr + 1;
        end
    end
    
    // Read logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_ptr <= 0;
            read_en <= 1'b0;
        end else if (read_en && !empty) begin
            spike_out <= spike_buf[read_ptr];
            channel_out <= channel_buf[read_ptr];
            x_out <= x_buf[read_ptr];
            y_out <= y_buf[read_ptr];
            read_ptr <= read_ptr + 1;
        end
    end
    
    // Count logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
        end else begin
            case ({write_en && !full, read_en && !empty})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end

endmodule