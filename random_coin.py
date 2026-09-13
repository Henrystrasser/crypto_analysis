import random

def munzwurf_simulation():
    print("Starte Münzwurf-Simulation (100 Würfe):")
    print("-" * 40)
    
    for i in range(1, 301):
        # random.randint(0, 1) wählt zufällig entweder 0 oder 1
        ergebnis = random.randint(0, 1)
        print(f"Wurf {i:3d}: {ergebnis}")
        
    print("-" * 40)
    print("Simulation beendet.")

if __name__ == "__main__":
    munzwurf_simulation()