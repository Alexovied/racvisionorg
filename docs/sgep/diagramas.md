# SGEP-Dashboard — Diagramas

Los diagramas están en Mermaid: GitHub y VS Code (con la extensión de Mermaid) los dibujan directamente; para exportarlos a imagen se puede pegar cada bloque en <https://mermaid.live>.

## Casos de uso

```mermaid
flowchart LR
    Admin(["👤 Admin"])
    PM(["👤 PM"])
    Auditor(["👤 Auditor"])
    Invitado(["👤 Stakeholder / invitado"])

    subgraph SGEP["SGEP-Dashboard"]
        UC1["Gestionar usuarios y roles"]
        UC2["ABM de proyectos de 7mo"]
        UC3["ABM de responsables"]
        UC4["Construir encuesta (drag & drop)"]
        UC5["Usar / guardar plantillas"]
        UC6["Definir lógica condicional y anonimato"]
        UC7["Invitar por email (link único)"]
        UC8["Compartir link general / QR"]
        UC9["Enviar recordatorios"]
        UC10["Abrir / cerrar encuesta"]
        UC11["Controlar quién respondió"]
        UC12["Ver dashboard con filtros"]
        UC13["Exportar PDF / Excel"]
        UC14["Ver registro de correos"]
        UC15["Responder encuesta"]
    end

    Admin --> UC1
    Admin --> UC2 & UC3 & UC4 & UC7 & UC12
    PM --> UC2 & UC3 & UC4 & UC5 & UC6 & UC7 & UC8 & UC9 & UC10 & UC11
    PM --> UC12 & UC13 & UC14
    Auditor --> UC11 & UC12 & UC13 & UC14
    Invitado --> UC15
    UC4 -. include .-> UC6
    UC4 -. extend .-> UC5
```

Admin hereda todos los casos de uso del PM (se dibujan solo algunos para no recargar el diagrama).

## Flujo: de la encuesta al dashboard

```mermaid
flowchart TD
    A([Inicio]) --> B[PM inicia sesión]
    B --> C{¿Existe el proyecto?}
    C -- No --> D[Cargar responsable y proyecto]
    D --> E
    C -- Sí --> E[Nueva encuesta: plantilla o en blanco]
    E --> F[Editor: arrastrar preguntas, condiciones, anonimato]
    F --> G{¿Preguntas válidas?}
    G -- No --> F
    G -- Sí --> H[Guardar encuesta]
    H --> I{¿Cómo se distribuye?}
    I -- Email --> J[Invitar: link único por persona y correo]
    I -- Stand / clase --> K[Link general + QR]
    J --> L[Stakeholder abre el link]
    K --> L
    L --> M{¿Encuesta abierta y link sin usar?}
    M -- No --> N[Mensaje: cerrada / ya respondiste]
    M -- Sí --> O[Responde; se muestran solo las preguntas que aplican]
    O --> P{¿Respuesta válida?}
    P -- No --> O
    P -- Sí --> Q[Guardar respuesta y marcar invitación]
    Q --> R[Dashboard se actualiza cada 10 s: KPIs, gráficos, semáforo]
    J --> S{¿Pendientes?}
    S -- Sí --> T[Enviar recordatorio] --> L
    R --> U[Exportar PDF / Excel]
    U --> V([Decisiones del equipo])
```

## Gantt del desarrollo

```mermaid
gantt
    title SGEP-Dashboard — Preparándonos para la Expotécnica
    dateFormat YYYY-MM-DD
    axisFormat %d/%m

    section Análisis
    Relevamiento de la problemática        :a1, 2026-08-03, 5d
    Casos de uso y requisitos              :a2, after a1, 4d
    section Diseño
    Modelo de datos y roles                :d1, after a2, 3d
    Diseño de pantallas y flujo            :d2, after a2, 4d
    section Desarrollo
    M1 Proyectos y responsables            :m1, after d2, 4d
    M2 Constructor de encuestas            :m2, after m1, 8d
    M3 Distribución y recolección          :m3, after m2, 6d
    M4 Dashboard dinámico                  :m4, after m3, 8d
    M5 Reportes y roles                    :m5, after m4, 5d
    section Calidad
    Tests de KPIs, roles y CSRF            :t1, after m3, 14d
    Datos de prueba y demo                 :t2, after m5, 3d
    section Documentación
    Manual de usuario y técnico            :doc, after m5, 5d
    section Cierre
    Ensayo de la presentación              :crit, e1, after doc, 3d
    Expotécnica                            :milestone, crit, after e1, 0d
```
