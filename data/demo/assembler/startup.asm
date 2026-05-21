; ---------------------------------------------------------------------------
; startup.asm - demo bootstrap routine for a small embedded controller.
; Demonstrates how the AsmChunker splits sources along labels.
; ---------------------------------------------------------------------------

        .section .text
        .global  _start

init:
        ; Initialize stack pointer and clear data segment.
        mov     sp, #0x8000
        mov     r0, #0
        mov     r1, #data_start
        mov     r2, #data_end
clear_loop:
        cmp     r1, r2
        bge     init_done
        strb    r0, [r1], #1
        b       clear_loop
init_done:
        bl      main_loop

main_loop:
        ; Poll the sensor and dispatch handlers.
        bl      read_sensor
        cmp     r0, #0
        beq     main_loop
        cmp     r0, #ERR_OVERFLOW
        beq     error_handler
        bl      process_value
        b       main_loop

read_sensor:
        ; Read a single sample into r0 from the ADC.
        ldr     r0, =ADC_DATA
        ldr     r0, [r0]
        bx      lr

process_value:
        ; Multiply by gain and store into buffer.
        ldr     r1, =GAIN
        ldr     r1, [r1]
        mul     r0, r0, r1
        ldr     r2, =buffer_ptr
        ldr     r3, [r2]
        str     r0, [r3], #4
        str     r3, [r2]
        bx      lr

error_handler:
        ; Log the error code and reset the controller.
        ldr     r1, =ERR_LOG
        str     r0, [r1]
        bl      reset_system
        b       init

reset_system:
        mov     r0, #0
        ldr     r1, =SYS_CTRL
        str     r0, [r1]
        bx      lr
