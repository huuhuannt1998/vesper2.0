/**
 * Sample IoT Sensor Firmware for VESPER QEMU Simulation
 * 
 * This is a minimal firmware that simulates a temperature/humidity sensor.
 * It can be compiled for ARM Cortex-M3 (LM3S6965) and run in QEMU.
 * 
 * Build with arm-none-eabi-gcc:
 *   arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -nostartfiles \
 *       -T linker.ld sensor_firmware.c -o sensor_firmware.elf
 * 
 * Or use the Makefile in this directory.
 */

#include <stdint.h>
#include <string.h>

/* UART registers for LM3S6965 */
#define UART0_BASE  0x4000C000
#define UART0_DR    (*(volatile uint32_t *)(UART0_BASE + 0x000))
#define UART0_FR    (*(volatile uint32_t *)(UART0_BASE + 0x018))
#define UART0_IBRD  (*(volatile uint32_t *)(UART0_BASE + 0x024))
#define UART0_FBRD  (*(volatile uint32_t *)(UART0_BASE + 0x028))
#define UART0_LCRH  (*(volatile uint32_t *)(UART0_BASE + 0x02C))
#define UART0_CTL   (*(volatile uint32_t *)(UART0_BASE + 0x030))

/* System control */
#define SYSCTL_RCGC1    (*(volatile uint32_t *)0x400FE104)
#define SYSCTL_RCGC2    (*(volatile uint32_t *)0x400FE108)

/* GPIO for LED */
#define GPIO_PORTF_BASE 0x40025000
#define GPIO_PORTF_DATA (*(volatile uint32_t *)(GPIO_PORTF_BASE + 0x3FC))
#define GPIO_PORTF_DIR  (*(volatile uint32_t *)(GPIO_PORTF_BASE + 0x400))
#define GPIO_PORTF_DEN  (*(volatile uint32_t *)(GPIO_PORTF_BASE + 0x51C))

/* Simulated sensor state */
static float temperature = 22.5f;
static float humidity = 45.0f;
static uint8_t led_state = 0;
static uint8_t motion_detected = 0;
static uint32_t tick_count = 0;

/* Random seed for simulation */
static uint32_t rand_seed = 12345;

/* Simple pseudo-random number generator */
static uint32_t simple_rand(void) {
    rand_seed = rand_seed * 1103515245 + 12345;
    return (rand_seed >> 16) & 0x7FFF;
}

/* Float to int with one decimal */
static int float_to_int10(float f) {
    return (int)(f * 10);
}

/* Initialize UART */
static void uart_init(void) {
    /* Enable UART0 clock */
    SYSCTL_RCGC1 |= (1 << 0);
    
    /* Wait for clock to stabilize */
    volatile int delay = 100;
    while (delay--);
    
    /* Disable UART */
    UART0_CTL = 0;
    
    /* Set baud rate: 115200 @ 12 MHz clock */
    UART0_IBRD = 6;   /* Integer part */
    UART0_FBRD = 33;  /* Fractional part */
    
    /* 8N1 configuration */
    UART0_LCRH = (0x3 << 5);  /* 8-bit word length */
    
    /* Enable UART */
    UART0_CTL = (1 << 0) | (1 << 8) | (1 << 9);  /* UARTEN, TXE, RXE */
}

/* Send a character via UART */
static void uart_putc(char c) {
    /* Wait until TX FIFO is not full */
    while (UART0_FR & (1 << 5));
    UART0_DR = c;
}

/* Send a string via UART */
static void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

/* Send a number as string */
static void uart_put_int(int n) {
    char buf[16];
    int i = 0;
    int neg = 0;
    
    if (n < 0) {
        neg = 1;
        n = -n;
    }
    
    do {
        buf[i++] = '0' + (n % 10);
        n /= 10;
    } while (n > 0);
    
    if (neg) {
        uart_putc('-');
    }
    
    while (i > 0) {
        uart_putc(buf[--i]);
    }
}

/* Send a float with one decimal */
static void uart_put_float1(float f) {
    int val = float_to_int10(f);
    int integer = val / 10;
    int decimal = val % 10;
    if (decimal < 0) decimal = -decimal;
    
    uart_put_int(integer);
    uart_putc('.');
    uart_putc('0' + decimal);
}

/* Receive a character (non-blocking) */
static int uart_getc(void) {
    if (UART0_FR & (1 << 4)) {
        return -1;  /* RX FIFO empty */
    }
    return UART0_DR & 0xFF;
}

/* Read a line from UART */
static int uart_readline(char *buf, int max_len) {
    int i = 0;
    int c;
    
    while (i < max_len - 1) {
        c = uart_getc();
        if (c < 0) {
            continue;  /* No data yet */
        }
        if (c == '\n' || c == '\r') {
            break;
        }
        buf[i++] = (char)c;
    }
    buf[i] = '\0';
    return i;
}

/* Initialize LED GPIO */
static void led_init(void) {
    /* Enable GPIO Port F clock */
    SYSCTL_RCGC2 |= (1 << 5);
    
    volatile int delay = 100;
    while (delay--);
    
    /* Configure PF0 as output (LED) */
    GPIO_PORTF_DIR |= (1 << 0);
    GPIO_PORTF_DEN |= (1 << 0);
}

/* Set LED state */
static void led_set(uint8_t on) {
    led_state = on;
    if (on) {
        GPIO_PORTF_DATA |= (1 << 0);
    } else {
        GPIO_PORTF_DATA &= ~(1 << 0);
    }
}

/* Update simulated sensor values */
static void update_sensors(void) {
    /* Add some random variation */
    int r = (int)simple_rand() % 100 - 50;  /* -50 to +49 */
    temperature += (float)r * 0.01f;
    
    /* Keep in realistic range */
    if (temperature < 15.0f) temperature = 15.0f;
    if (temperature > 35.0f) temperature = 35.0f;
    
    r = (int)simple_rand() % 100 - 50;
    humidity += (float)r * 0.05f;
    if (humidity < 20.0f) humidity = 20.0f;
    if (humidity > 80.0f) humidity = 80.0f;
    
    /* Random motion events */
    if ((simple_rand() % 1000) < 5) {
        motion_detected = 1;
    }
}

/* Process a command */
static void process_command(const char *cmd) {
    if (strcmp(cmd, "GET_TEMP") == 0) {
        uart_puts("TEMP:");
        uart_put_float1(temperature);
        uart_puts("\n");
    }
    else if (strcmp(cmd, "GET_HUMIDITY") == 0) {
        uart_puts("HUMIDITY:");
        uart_put_float1(humidity);
        uart_puts("\n");
    }
    else if (strcmp(cmd, "GET_ALL") == 0) {
        uart_puts("TEMP:");
        uart_put_float1(temperature);
        uart_puts("\n");
        uart_puts("HUMIDITY:");
        uart_put_float1(humidity);
        uart_puts("\n");
        uart_puts("LED:");
        uart_put_int(led_state);
        uart_puts("\n");
    }
    else if (strcmp(cmd, "SET_LED:1") == 0 || strcmp(cmd, "SET_LED:ON") == 0) {
        led_set(1);
        uart_puts("LED:1\n");
        uart_puts("ACK:SET_LED\n");
    }
    else if (strcmp(cmd, "SET_LED:0") == 0 || strcmp(cmd, "SET_LED:OFF") == 0) {
        led_set(0);
        uart_puts("LED:0\n");
        uart_puts("ACK:SET_LED\n");
    }
    else if (strcmp(cmd, "STATUS") == 0) {
        uart_puts("STATUS:OK\n");
    }
    else if (strcmp(cmd, "IDENTIFY") == 0) {
        uart_puts("DEVICE:VESPER_SENSOR_V1\n");
        uart_puts("TYPE:TEMPERATURE_HUMIDITY\n");
        uart_puts("FIRMWARE:1.0.0\n");
    }
    else if (strcmp(cmd, "REBOOT") == 0) {
        uart_puts("ACK:REBOOT\n");
        /* In real firmware, this would trigger a reset */
        uart_puts("BOOTED\n");
        uart_puts("READY\n");
    }
    else if (cmd[0] != '\0') {
        uart_puts("ERROR:UNKNOWN_CMD:");
        uart_puts(cmd);
        uart_puts("\n");
    }
}

/* Simple delay */
static void delay(volatile uint32_t count) {
    while (count--);
}

/* Main function */
int main(void) {
    char cmd_buf[64];
    int cmd_len;
    
    /* Initialize peripherals */
    uart_init();
    led_init();
    
    /* Boot message */
    uart_puts("BOOTED\n");
    uart_puts("DEVICE:VESPER_SENSOR_V1\n");
    uart_puts("READY\n");
    
    /* Main loop */
    while (1) {
        /* Check for commands */
        cmd_len = 0;
        int c;
        int timeout = 10000;
        
        while (timeout-- > 0) {
            c = uart_getc();
            if (c >= 0) {
                if (c == '\n' || c == '\r') {
                    cmd_buf[cmd_len] = '\0';
                    if (cmd_len > 0) {
                        process_command(cmd_buf);
                    }
                    cmd_len = 0;
                    timeout = 10000;  /* Reset timeout */
                } else if (cmd_len < sizeof(cmd_buf) - 1) {
                    cmd_buf[cmd_len++] = (char)c;
                }
            }
        }
        
        /* Update sensors periodically */
        tick_count++;
        if (tick_count % 100 == 0) {
            update_sensors();
        }
        
        /* Check for motion event */
        if (motion_detected) {
            uart_puts("EVENT:MOTION\n");
            motion_detected = 0;
        }
        
        /* Small delay */
        delay(1000);
    }
    
    return 0;
}

/* Exception handlers (required for bare-metal) */
void Reset_Handler(void) __attribute__((weak, alias("main")));
void NMI_Handler(void) { while (1); }
void HardFault_Handler(void) { while (1); }
void MemManage_Handler(void) { while (1); }
void BusFault_Handler(void) { while (1); }
void UsageFault_Handler(void) { while (1); }
void SVC_Handler(void) { while (1); }
void DebugMon_Handler(void) { while (1); }
void PendSV_Handler(void) { while (1); }
void SysTick_Handler(void) { while (1); }
